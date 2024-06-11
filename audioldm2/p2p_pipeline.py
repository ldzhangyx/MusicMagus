# Copyright 2023 CVSSP, ByteDance and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.models.attention_processor import Attention
from audioldm2.utils import wav_to_fbank, TacotronSTFT
import re

if is_librosa_available():
    import librosa

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CURRENT_LOSS = None
CURRENT_TIMESTEP = None

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
"""


def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


def prepare_unet(unet: AudioLDM2UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True)
            module.requires_grad_(True)
        else:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=False)
            module.requires_grad_(False)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet


class Pix2PixZeroL2Loss:
    def __init__(self):
        self.loss = 0.0

    def compute_loss(self, predictions, targets):

        self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)


class Pix2PixZeroMSELoss:
    def __init__(self, seq_len, index, offset=1, weights=1.0, device="cuda"):
        self.loss = 0.0
        self.device = device
        self.seq_len = seq_len
        self.weights = torch.ones(seq_len, device=self.device)
        self.weights[index: index + offset] = weights

    def compute_loss(self, predictions, targets):

        # try
        # predictions = torch.nn.functional.avg_pool2d(predictions, kernel_size=2, stride=2)
        # targets = torch.nn.functional.avg_pool2d(targets, kernel_size=2, stride=2)

        loss = F.mse_loss(predictions, targets, reduction='none')
        if self.seq_len == loss.shape[-1]:  # we only re-weight a part of cross-attention maps.
            loss = loss * self.weights.unsqueeze(0).unsqueeze(1)
        self.loss += loss.sum((1, 2)).mean(0)


class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        residual = hidden_states

        # batch_size, sequence_length, _ = hidden_states.shape
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            self.is_pix2pix_zero = False  # no cross-attention.
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.is_pix2pix_zero and CURRENT_TIMESTEP is not None:
            # new bookkeeping to save the attention weights.
            if CURRENT_LOSS is None:
                self.reference_cross_attn_map[CURRENT_TIMESTEP.item()] = attention_probs.detach().cpu()
            # compute loss
            elif CURRENT_LOSS is not None:
                prev_attn_probs = self.reference_cross_attn_map.pop(CURRENT_TIMESTEP.item())
                CURRENT_LOSS.compute_loss(attention_probs, prev_attn_probs.to(attention_probs.device))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AudioLDM2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.ClapModel`]):
            First frozen text-encoder. AudioLDM2 uses the joint audio-text embedding model
            [CLAP](https://huggingface.co/docs/transformers/model_doc/clap#transformers.CLAPTextModelWithProjection),
            specifically the [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant. The
            text branch is used to encode the text prompt to a prompt embedding. The full audio-text model is used to
            rank generated waveforms against the text prompt by computing similarity scores.
        text_encoder_2 ([`~transformers.T5EncoderModel`]):
            Second frozen text-encoder. AudioLDM2 uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) variant.
        projection_model ([`AudioLDM2ProjectionModel`]):
            A trained model used to linearly project the hidden-states from the first and second text encoder models
            and insert learned SOS and EOS token embeddings. The projected hidden-states from the two text encoders are
            concatenated to give the input to the language model.
        language_model ([`~transformers.GPT2Model`]):
            An auto-regressive language model used to generate a sequence of hidden-states conditioned on the projected
            outputs from the two text encoders.
        tokenizer ([`~transformers.RobertaTokenizer`]):
            Tokenizer to tokenize text for the first frozen text-encoder.
        tokenizer_2 ([`~transformers.T5Tokenizer`]):
            Tokenizer to tokenize text for the second frozen text-encoder.
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            Feature extractor to pre-process generated audio waveforms to log-mel spectrograms for automatic scoring.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            Vocoder of class `SpeechT5HifiGan` to convert the mel-spectrogram latents to the final audio waveform.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapModel,
        text_encoder_2: T5EncoderModel,
        projection_model: AudioLDM2ProjectionModel,
        language_model: GPT2Model,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast],
        feature_extractor: ClapFeatureExtractor,
        unet: AudioLDM2UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = [
            self.text_encoder.text_model,
            self.text_encoder.text_projection,
            self.text_encoder_2,
            self.projection_model,
            self.language_model,
            self.unet,
            self.vae,
            self.vocoder,
            self.text_encoder,
        ]

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        fix_length: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed text embeddings from the Flan T5 model. Can be used to easily tweak text inputs, *e.g.*
                prompt weighting. If not provided, text embeddings will be computed from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed negative text embeddings from the Flan T5 model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
        Returns:
            prompt_embeds (`torch.FloatTensor`):
                Text embeddings from the Flan T5 model.
            attention_mask (`torch.LongTensor`):
                Attention mask to be applied to the `prompt_embeds`.
            generated_prompt_embeds (`torch.FloatTensor`):
                Text embeddings generated from the GPT2 langauge model.

        Example:

        ```python
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get text embedding vectors
        >>> prompt_embeds, attention_mask, generated_prompt_embeds = pipe.encode_prompt(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass text embeddings to pipeline for text-conditional audio generation
        >>> audio = pipe(
        ...     prompt_embeds=prompt_embeds,
        ...     attention_mask=attention_mask,
        ...     generated_prompt_embeds=generated_prompt_embeds,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ... ).audios[0]

        >>> # save generated audio sample
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```"""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]



        if prompt_embeds is None:
            prompt_embeds_list = []
            attention_mask_list = []

            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length" if (isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) or fix_length) else True,
                    max_length=fix_length if fix_length is not None else tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                attention_mask = text_inputs.attention_mask
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    # logger.warning(
                    #     f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                    #     f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                    # )

                text_input_ids = text_input_ids.to(device)
                attention_mask = attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    prompt_embeds = prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    attention_mask = attention_mask.new_ones((batch_size, 1))
                else:
                    prompt_embeds = text_encoder(
                        text_input_ids,
                        attention_mask=attention_mask,
                    )
                    prompt_embeds = prompt_embeds[0]

                prompt_embeds_list.append(prompt_embeds)
                attention_mask_list.append(attention_mask)

            projection_output = self.projection_model(
                hidden_states=prompt_embeds_list[0],
                hidden_states_1=prompt_embeds_list[1],
                attention_mask=attention_mask_list[0],
                attention_mask_1=attention_mask_list[1],
            )
            projected_prompt_embeds = projection_output.hidden_states
            projected_attention_mask = projection_output.attention_mask

            generated_prompt_embeds = self.generate_language_model(
                projected_prompt_embeds,
                attention_mask=projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        attention_mask = (
            attention_mask.to(device=device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(dtype=self.language_model.dtype, device=device)

        bs_embed, seq_len, hidden_size = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, hidden_size)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_waveforms_per_prompt, seq_len)

        bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
        # duplicate generated embeddings for each generation per prompt, using mps friendly method
        generated_prompt_embeds = generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        generated_prompt_embeds = generated_prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            negative_attention_mask_list = []
            max_length = prompt_embeds.shape[1]
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=tokenizer.model_max_length
                    if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
                    else max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                uncond_input_ids = uncond_input.input_ids.to(device)
                negative_attention_mask = uncond_input.attention_mask.to(device)

                if text_encoder.config.model_type == "clap":
                    negative_prompt_embeds = text_encoder.get_text_features(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                    negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                    # make sure that we attend to this single hidden-state
                    negative_attention_mask = negative_attention_mask.new_ones((batch_size, 1))
                else:
                    negative_prompt_embeds = text_encoder(
                        uncond_input_ids,
                        attention_mask=negative_attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

                negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_attention_mask_list.append(negative_attention_mask)

            projection_output = self.projection_model(
                hidden_states=negative_prompt_embeds_list[0],
                hidden_states_1=negative_prompt_embeds_list[1],
                attention_mask=negative_attention_mask_list[0],
                attention_mask_1=negative_attention_mask_list[1],
            )
            negative_projected_prompt_embeds = projection_output.hidden_states
            negative_projected_attention_mask = projection_output.attention_mask

            negative_generated_prompt_embeds = self.generate_language_model(
                negative_projected_prompt_embeds,
                attention_mask=negative_projected_attention_mask,
                max_new_tokens=max_new_tokens,
            )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_attention_mask = (
                negative_attention_mask.to(device=device)
                if negative_attention_mask is not None
                else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
            )
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
                dtype=self.language_model.dtype, device=device
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            negative_attention_mask = negative_attention_mask.repeat(1, num_waveforms_per_prompt)
            negative_attention_mask = negative_attention_mask.view(batch_size * num_waveforms_per_prompt, seq_len)

            # duplicate unconditional generated embeddings for each generation per prompt
            seq_len = negative_generated_prompt_embeds.shape[1]
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_generated_prompt_embeds = negative_generated_prompt_embeds.view(
                batch_size * num_waveforms_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask])
            generated_prompt_embeds = torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds])

        return prompt_embeds, attention_mask, generated_prompt_embeds

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
        if not is_librosa_available():
            logger.info(
                "Automatic scoring of the generated audio waveforms against the input prompt text requires the "
                "`librosa` package to resample the generated waveforms. Returning the audios in the order they were "
                "generated. To enable automatic scoring, install `librosa` with: `pip install librosa`."
            )
            return audio
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        resampled_audio = librosa.resample(
            audio.numpy(), orig_sr=self.vocoder.config.sampling_rate, target_sr=self.feature_extractor.sampling_rate
        )
        inputs["input_features"] = self.feature_extractor(
            list(resampled_audio), return_tensors="pt", sampling_rate=self.feature_extractor.sampling_rate
        ).input_features.type(dtype)
        inputs = inputs.to(device)

        # compute the audio-text similarity score using the CLAP model
        logits_per_text = self.text_encoder(**inputs).logits_per_text
        # sort by the highest matching generations per prompt
        indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
        audio = torch.index_select(audio, 0, indices.reshape(-1).cpu())
        return audio

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        generated_prompt_embeds=None,
        negative_generated_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and (prompt_embeds is None or generated_prompt_embeds is None):
            raise ValueError(
                "Provide either `prompt`, or `prompt_embeds` and `generated_prompt_embeds`. Cannot leave "
                "`prompt` undefined without specifying both `prompt_embeds` and `generated_prompt_embeds`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_embeds is not None and negative_generated_prompt_embeds is None:
            raise ValueError(
                "Cannot forward `negative_prompt_embeds` without `negative_generated_prompt_embeds`. Ensure that"
                "both arguments are specified"
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if attention_mask is not None and attention_mask.shape != prompt_embeds.shape[:2]:
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {attention_mask.shape} != `prompt_embeds` {prompt_embeds.shape}"
                )

        if generated_prompt_embeds is not None and negative_generated_prompt_embeds is not None:
            if generated_prompt_embeds.shape != negative_generated_prompt_embeds.shape:
                raise ValueError(
                    "`generated_prompt_embeds` and `negative_generated_prompt_embeds` must have the same shape when "
                    f"passed directly, but got: `generated_prompt_embeds` {generated_prompt_embeds.shape} != "
                    f"`negative_generated_prompt_embeds` {negative_generated_prompt_embeds.shape}."
                )
            if (
                negative_attention_mask is not None
                and negative_attention_mask.shape != negative_prompt_embeds.shape[:2]
            ):
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {negative_attention_mask.shape} != `prompt_embeds` {negative_prompt_embeds.shape}"
                )

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """Constructs the edit direction to steer the image generation process semantically."""
        return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_generated_prompt_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
        # for p2p
        source_embeddings=None,
        generated_source_embeddings=None,
        target_embeddings=None,
        generated_target_embeddings=None,
        cross_attention_guidance_amount=0.15,
        before_concept=None,
        after_concept=None,
        strategy=None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
            audio_length_in_s (`int`, *optional*, defaults to 10.24):
                The length of the generated audio sample in seconds.
            num_inference_steps (`int`, *optional*, defaults to 200):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                A higher guidance scale value encourages the model to generate audio that is closely linked to the text
                `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt. If `num_waveforms_per_prompt > 1`, then automatic
                scoring is performed between the generated outputs and the text prompt. This scoring ranks the
                generated waveforms based on their cosine similarity with the text input in the joint text-audio
                embedding space.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for spectrogram
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings from the GPT2 langauge model. Can be used to easily tweak text inputs,
                 *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input
                 argument.
            negative_generated_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings from the GPT2 language model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_prompt_embeds`. If not provided, attention
                mask will be computed from `negative_prompt` input argument.
            max_new_tokens (`int`, *optional*, defaults to None):
                Number of new tokens to generate with the GPT2 language model. If not provided, number of tokens will
                be taken from the config of the model.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or
                `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion
                model (LDM) output.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated audio.
        """
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        global CURRENT_TIMESTEP
        global CURRENT_LOSS

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            generated_prompt_embeds,
            negative_generated_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        target_prompt = re.sub(re.escape(before_concept), after_concept, prompt, flags=re.IGNORECASE)
        # target_prompt = prompt.replace(before_concept, after_concept)
        replace_index = target_prompt.index(after_concept)
        replace_offset = len(after_concept)  # not as good as token-level replacement but still works.

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generated_prompt_embeds=generated_prompt_embeds,
            negative_generated_prompt_embeds=negative_generated_prompt_embeds,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_init = latents.clone()

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # p2p-1. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # It is difficult to pass the timestep param. Using global variable
                CURRENT_TIMESTEP = t
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                    cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # decode
        latents_before = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram_before = self.vae.decode(latents_before).sample
        audio_before = self.mel_to_audio_with_scoring(device, mel_spectrogram_before, num_waveforms_per_prompt,
                                                      original_waveform_length, prompt, prompt_embeds)



        self.maybe_free_model_hooks()


        if strategy == "swap":
            if output_type == "np":
                audio_before = audio_before.numpy()

            if not return_dict:
                return (audio_before,), None

            return AudioPipelineOutput(audios=audio_before), None


        # p2p-2. Compute the edit directions.
        edit_direction = 1 * self.construct_direction(source_embeddings, target_embeddings).to(prompt_embeds.device)
        # edit_direction_1 = 1.15 * self.construct_direction(generated_source_embeddings, generated_target_embeddings).to(prompt_embeds.device)
        #
        #
        # p2p-3. Edit the prompt embeddings as per the edit directions discovered.
        prompt_embeds_edit = prompt_embeds.clone()
        if prompt_embeds.shape[1] != edit_direction.shape[1]:
            print(f"Warning: edit direction {edit_direction.shape[1]} truncated to match prompt length {prompt_embeds_edit.shape[1]}.")
            if edit_direction.shape[1] > prompt_embeds.shape[1]: # truncate
                edit_direction = edit_direction[:, :prompt_embeds.shape[1], :]
            else: # padding
                edit_direction = torch.cat([edit_direction, torch.zeros([edit_direction.shape[0], prompt_embeds_edit.shape[1] - edit_direction.shape[1], edit_direction.shape[2]], device=edit_direction.device)], dim=1)
        prompt_embeds_edit[1:2] += edit_direction  #
        # generated_prompt_embeds_edit = generated_prompt_embeds.clone()
        # generated_prompt_embeds_edit[1:2] += edit_direction_1

        # if negative_prompt is None:
        #     negative_prompt = ""

        _, _, generated_prompt_embeds_edit = self.encode_prompt(
            target_prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # p2p-4. Second denoising loop to generate the edited image.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = latents_init
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # we want to learn the latent such that it steers the generation
                # process towards the edited direction, so make the make initial
                # noise learnable
                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True

                # time-related dynamic guidance amount
                # cross_attention_guidance_amount = cross_attention_guidance_amount * torch.tanh(torch.tensor(t / num_inference_steps))

                # optimizer
                opt = torch.optim.SGD([x_in], lr=cross_attention_guidance_amount)

                with torch.enable_grad():
                    # initialize loss
                    CURRENT_LOSS = Pix2PixZeroMSELoss(
                        seq_len=prompt_embeds_edit.shape[1],
                        index=replace_index,
                        offset=replace_offset,
                        weights=1,  # remember the weight
                        device="cuda"
                    )
                    CURRENT_TIMESTEP = t
                    # predict the noise residual
                    noise_pred = self.unet(
                        x_in,
                        t,
                        encoder_hidden_states=generated_prompt_embeds_edit.detach(),
                        encoder_hidden_states_1=prompt_embeds_edit.detach(),
                        encoder_attention_mask_1=attention_mask,
                        cross_attention_kwargs={"timestep": CURRENT_TIMESTEP, "loss": CURRENT_LOSS},
                    )[0]

                    if cross_attention_guidance_amount > 0:
                        CURRENT_LOSS.loss.backward(retain_graph=False)
                        opt.step()

                if cross_attention_guidance_amount > 0:
                    # recompute the noise
                    CURRENT_TIMESTEP = None
                    noise_pred = self.unet(
                        x_in.detach(),
                        t,
                        encoder_hidden_states=generated_prompt_embeds_edit,
                        encoder_hidden_states_1=prompt_embeds_edit,
                        encoder_attention_mask_1=attention_mask,
                        cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                    )[0]

                latents = x_in.detach().chunk(2)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        CURRENT_TIMESTEP, CURRENT_LOSS = None, None

        # 8. Post-processing
        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            mel_spectrogram = self.vae.decode(latents).sample
        else:
            return AudioPipelineOutput(audios=latents_before), AudioPipelineOutput(audios=latents)

        audio = self.mel_to_audio_with_scoring(device, mel_spectrogram, num_waveforms_per_prompt,
                                               original_waveform_length, target_prompt, prompt_embeds)

        if output_type == "np":
            audio = audio.numpy()
            audio_before = audio_before.numpy()

        if not return_dict:
            return (audio_before, ), (audio,)

        return AudioPipelineOutput(audios=audio_before), AudioPipelineOutput(audios=audio)

    def mel_to_audio_with_scoring(self, device, mel_spectrogram, num_waveforms_per_prompt, original_waveform_length,
                                  prompt, prompt_embeds):
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        # 9. Automatic scoring
        if num_waveforms_per_prompt > 1 and prompt is not None:
            audio = self.score_waveforms(
                text=prompt,
                audio=audio,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
            )
        return audio

    def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
        pred_type = self.inverse_scheduler.config.prediction_type
        alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output
        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

    def auto_corr_loss(self, hidden_states, generator=None):
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            for j in range(hidden_states.shape[1]):
                noise = hidden_states[i : i + 1, j : j + 1, :, :]
                while True:
                    roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2

                    # if noise.shape[2] <= 8:
                    #     break
                    break  # We first consider full latent
                    noise = F.avg_pool2d(noise, kernel_size=2)  # it is the problem
        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)


    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[str] = None,
        audio_path: str = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_corr_rolls: int = 5,
    ):
        r"""
        Function used to generate inverted latents given a prompt and image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch which will be used for conditioning. Can also accept
                image latents as `image`, if passing latents directly, it will not be encoded again.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            cross_attention_guidance_amount (`float`, defaults to 0.1):
                Amount of guidance needed from the reference cross-attention maps.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback–Leibler divergence output
            num_reg_steps (`int`, *optional*, defaults to 5):
                Number of regularization loss steps
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] or
            `tuple`:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is the inverted
            latents tensor and then second is the corresponding decoded image.
        """

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        audio_length_in_s = audio_16k.shape[1] / 16000
        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        fn_STFT = TacotronSTFT(
            1024,  # config["preprocessing"]["stft"]["filter_length"],
            160,  # config["preprocessing"]["stft"]["hop_length"],
            1024,  # config["preprocessing"]["stft"]["win_length"],
            64,  # config["preprocessing"]["mel"]["n_mel_channels"],
            16000,  # config["preprocessing"]["audio"]["sampling_rate"],
            0,  # config["preprocessing"]["mel"]["mel_fmin"],
            8000,  # config["preprocessing"]["mel"]["mel_fmax"],
        )

        mel, _, _ = wav_to_fbank(
            audio_path, target_length=int(audio_length_in_s * 102.4), fn_STFT=fn_STFT
        )
        mel = mel.unsqueeze(0).unsqueeze(0).to(device)

        print(mel.shape)

        audio_test = self.mel_to_audio_with_scoring(device, mel.cuda(), 1,
                                                      original_waveform_length, prompt, prompt_embeds)

        # 4. Prepare latent variables
        latents = self.vae.to(torch.float32).encode(mel.cuda()).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents

        latents_origin = latents.clone()

        # 5. Encode input prompt
        num_waveforms_per_prompt = 1
        prompt_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )


        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet).to(torch.float32)

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                    cross_attention_kwargs={"timestep": CURRENT_TIMESTEP},
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # regularization of the noise prediction
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if lambda_auto_corr > 0:
                            for _ in range(num_auto_corr_rolls):
                                var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                                # Derive epsilon from model output before regularizing to IID standard normal
                                var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                                l_ac = self.auto_corr_loss(var_epsilon, generator=generator)
                                l_ac.backward()

                                grad = var.grad.detach() / num_auto_corr_rolls
                                noise_pred = noise_pred - lambda_auto_corr * grad

                        if lambda_kl > 0:
                            var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                            l_kld = self.kl_divergence(var_epsilon)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - lambda_kl * grad

                        noise_pred = noise_pred.detach()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        inverted_latents = latents.detach().clone()

        # 8. Post-processing
        latents_before = 1 / self.vae.config.scaling_factor * latents_origin
        mel_spectrogram_before = self.vae.decode(latents_before).sample
        audio_before = self.mel_to_audio_with_scoring(device, mel_spectrogram_before, num_waveforms_per_prompt,
                                                      original_waveform_length, prompt, prompt_embeds)

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "np":
            audio_before = audio_before.numpy()

        return inverted_latents, audio_test


    @torch.no_grad()
    def style_transfer(
        self,
        prompt: Optional[str] = None,
        audio_path: str = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        audio_length_in_s = audio_16k.shape[1] / 16000
        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)

        fn_STFT = TacotronSTFT(
            1024,  # config["preprocessing"]["stft"]["filter_length"],
            160,  # config["preprocessing"]["stft"]["hop_length"],
            1024,  # config["preprocessing"]["stft"]["win_length"],
            64,  # config["preprocessing"]["mel"]["n_mel_channels"],
            16000,  # config["preprocessing"]["audio"]["sampling_rate"],
            0,  # config["preprocessing"]["mel"]["mel_fmin"],
            8000,  # config["preprocessing"]["mel"]["mel_fmax"],
        )

        mel, _, _ = wav_to_fbank(
            audio_path, target_length=int(audio_length_in_s * 102.4), fn_STFT=fn_STFT
        )
        mel = mel.unsqueeze(0).unsqueeze(0).to(device)

        print(mel.shape)

        audio_test = self.mel_to_audio_with_scoring(device, mel.cuda(), 1,
                                                      original_waveform_length, prompt, prompt_embeds)

        # 4. Prepare latent variables
        latents = self.vae.to(torch.float32).encode(mel.cuda()).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents

        latents_origin = latents.clone()

        # 5. Encode input prompt
        num_waveforms_per_prompt = 1
        prompt_embeds, attention_mask, generated_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )


        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 6. Rejig the UNet so that we can obtain the cross-attenion maps and
        # use them for guiding the subsequent image generation.
        self.unet = prepare_unet(self.unet).to(torch.float32)

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if i == int(len(timesteps) / 2):
                    break

        inverted_latents = latents.detach().clone()

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance

                if i < int(len(timesteps) / 2):
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # It is difficult to pass the timestep param. Using global variable
                CURRENT_TIMESTEP = t
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=generated_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=attention_mask,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 3.5 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        audio = self.mel_to_audio_with_scoring(device, mel_spectrogram, num_waveforms_per_prompt,
                                                      original_waveform_length, prompt, prompt_embeds)

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "np":
            audio = audio.numpy()

        return AudioPipelineOutput(audios=audio)