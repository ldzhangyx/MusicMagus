from diffusers import DDIMScheduler, DiffusionPipeline
import torch
import openai
import os

openai.api_key = ""

class EmbeddingCalculator(object):
    def __init__(self, embedding_model=None, prompt_length=10):
        if embedding_model is None:
            embedding_model = DiffusionPipeline.from_pretrained("cvssp/audioldm2-large")
        self.embedding_model = embedding_model.to("cuda")
        self.embedding_tokenizer = embedding_model.tokenizer
        self.prompt_length = prompt_length

    @staticmethod
    def generate_captions(input_prompt, number=24):
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=input_prompt,
            temperature=1.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=number,
        )
        return [item.text.strip() for item in response.choices]

    @staticmethod
    def postprocessing_caption(caption, keyword):
        input_prompt = (f"Please shorten the music caption, and include genre, main instrument, and mood. "
                        f"Then, please replace the main instrument to {keyword}. Caption: {caption}. Result: ")
        output = EmbeddingCalculator.generate_captions(input_prompt, number=1)
        return output[0]

    @torch.no_grad()
    def embed_captions(self, l_sentences, fix_length=None):
        prompt_embeds, _, generated_prompt_embeds = self.embedding_model.encode_prompt(
            prompt=l_sentences,
            device="cuda",
            do_classifier_free_guidance=False,
            num_waveforms_per_prompt=1,
            fix_length=fix_length,
        )

        # average
        prompt_embeds = prompt_embeds.mean(dim=0, keepdim=True)
        generated_prompt_embeds = generated_prompt_embeds.mean(dim=0, keepdim=True)

        return prompt_embeds, generated_prompt_embeds

    def __call__(self, source_concept, target_concept, verbose=True):
        source_text = (f"Generate one sentence capturing keywords of {source_concept} music around {self.prompt_length} words. "
                       f"Answer: ")
        target_text = source_text.replace(source_concept, target_concept)

        # text -> captions
        source_captions = self.generate_captions(source_text)
        target_captions = self.generate_captions(target_text)

        if verbose:
            print(f"source: {source_captions}")
            print(f"target: {target_captions}")

        fix_length = self.prompt_length + 2

        # captions -> embeddings
        source_embeddings, generated_source_embeddings = self.embed_captions(source_captions, fix_length=fix_length)
        target_embeddings, generated_target_embeddings = self.embed_captions(target_captions, fix_length=fix_length)

        return source_embeddings, generated_source_embeddings, target_embeddings, generated_target_embeddings