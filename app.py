import gradio as gr
import random
import torch
from torch import inference_mode
# from tempfile import NamedTemporaryFile
from typing import Optional
import numpy as np
from models import load_model
import utils
import spaces
from inversion_utils import inversion_forward_process, inversion_reverse_process







# ===

intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;"> MusicMagus </h1>
<h2 style="font-weight: 1400; text-align: center; margin-bottom: 7px;"> Zero-Shot Text-to-Music Editing via Diffusion Models </h2>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2402.06178">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://wry-neighbor-173.notion.site/MusicMagus-Zero-Shot-Text-to-Music-Editing-via-Diffusion-Models-8f55a82f34944eb9a4028ca56c546d9d">[Demo page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/ldzhangyx/MusicMagus">[Code]</a>
</h3>

"""

help = """
<div style="font-size:medium">
<b>Instructions:</b><br>
<ul style="line-height: normal">
<li>You must provide an input audio and a target prompt to edit the audio. </li>
<li>T<sub>start</sub> is used to control the tradeoff between fidelity to the original signal and text-adhearance.
Lower value -> favor fidelity. Higher value -> apply a stronger edit.</li>
<li>Make sure that you use an AudioLDM2 version that is suitable for your input audio.
For example, use the music version for music and the large version for general audio.
</li>x
<li>You can additionally provide a source prompt to guide even further the editing process.</li>
<li>Longer input will take more time.</li>
<li><strong>Unlimited length</strong>: This space automatically trims input audio to a maximum length of 30 seconds.
For unlimited length, duplicated the space, and remove the trimming by changing the code.
Specifically, in the <code style="display:inline; background-color: lightgrey; ">load_audio</code> function in the <code style="display:inline; background-color: lightgrey; ">utils.py</code> file,
change <code style="display:inline; background-color: lightgrey; ">duration = min(audioldm.utils.get_duration(audio_path), 30)</code> to 
<code style="display:inline; background-color: lightgrey; ">duration = audioldm.utils.get_duration(audio_path)</code>.
</ul>
</div>

"""

with gr.Blocks(css='style.css') as demo:  #, delete_cache=(3600, 3600)) as demo:
    def reset_do_inversion(do_inversion_user, do_inversion):
        # do_inversion = gr.State(value=True)
        do_inversion = True
        do_inversion_user = True
        return do_inversion_user, do_inversion

    # handle the case where the user clicked the button but the inversion was not done
    def clear_do_inversion_user(do_inversion_user):
        do_inversion_user = False
        return do_inversion_user

    def post_match_do_inversion(do_inversion_user, do_inversion):
        if do_inversion_user:
            do_inversion = True
            do_inversion_user = False
        return do_inversion_user, do_inversion

    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    wtszs = gr.State()
    # cache_dir = gr.State(demo.GRADIO_CACHE)
    saved_inv_model = gr.State()
    # current_loaded_model = gr.State(value="cvssp/audioldm2-music")
    # ldm_stable = load_model("cvssp/audioldm2-music", device, 200)
    # ldm_stable = gr.State(value=ldm_stable)
    do_inversion = gr.State(value=True)  # To save some runtime when editing the same thing over and over
    do_inversion_user = gr.State(value=False)

    with gr.Group():
        gr.Markdown("💡 **note**: input longer than **30 sec** is automatically trimmed (for unlimited input, see the Help section below)")
        with gr.Row():
            input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", editable=True, label="Input Audio",
                                   interactive=True, scale=1)
            output_audio = gr.Audio(label="Edited Audio", interactive=False, scale=1)

    with gr.Row():
        tar_prompt = gr.Textbox(label="Prompt", info="Describe your desired edited output",
                                placeholder="a recording of a happy upbeat arcade game soundtrack",
                                lines=2, interactive=True)

    with gr.Row():
        t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label="T-start (%)", interactive=True, scale=3,
                            info="Lower T-start -> closer to original audio. Higher T-start -> stronger edit.")
        # model_id = gr.Radio(label="AudioLDM2 Version",
        model_id = gr.Dropdown(label="AudioLDM2 Version",
                               choices=["cvssp/audioldm2",
                                        "cvssp/audioldm2-large",
                                        "cvssp/audioldm2-music"],
                               info="Choose a checkpoint suitable for your intended audio and edit",
                               value="cvssp/audioldm2-music", interactive=True, type="value", scale=2)

    with gr.Row():
        with gr.Column():
            submit = gr.Button("Edit")

    with gr.Accordion("More Options", open=False):
        with gr.Row():
            src_prompt = gr.Textbox(label="Source Prompt", lines=2, interactive=True,
                                    info="Optional: Describe the original audio input",
                                    placeholder="A recording of a happy upbeat classical music piece",)

        with gr.Row():
            cfg_scale_src = gr.Number(value=3, minimum=0.5, maximum=25, precision=None,
                                      label="Source Guidance Scale", interactive=True, scale=1)
            cfg_scale_tar = gr.Number(value=12, minimum=0.5, maximum=25, precision=None,
                                      label="Target Guidance Scale", interactive=True, scale=1)
            steps = gr.Number(value=50, step=1, minimum=20, maximum=300,
                              info="Higher values (e.g. 200) yield higher-quality generation.",
                              label="Num Diffusion Steps", interactive=True, scale=1)
        with gr.Row():
            seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
            randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
            length = gr.Number(label="Length", interactive=False, visible=False)

    with gr.Accordion("Help💡", open=False):
        gr.HTML(help)

    submit.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=[seed], queue=False).then(
            fn=clear_do_inversion_user, inputs=[do_inversion_user], outputs=[do_inversion_user]).then(
           fn=edit,
           inputs=[#cache_dir,
                   input_audio,
                   model_id,
                   do_inversion,
                   #    current_loaded_model, ldm_stable,
                      wts, zs,
                #    wtszs,
                   saved_inv_model,
                   src_prompt,
                   tar_prompt,
                   steps,
                   cfg_scale_src,
                   cfg_scale_tar,
                   t_start,
                   randomize_seed
                   ],
           outputs=[output_audio, wts, zs, # wtszs,
                    saved_inv_model, do_inversion]  # , current_loaded_model, ldm_stable],
        ).then(post_match_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion]
               ).then(lambda x: (demo.temp_file_sets.append(set([str(gr.utils.abspath(x))])) if type(x) is str else None),
                      inputs=wtszs)

    # demo.move_resource_to_block_cache(wtszs.value)

    # If sources changed we have to rerun inversion
    input_audio.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    src_prompt.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    model_id.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    cfg_scale_src.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    steps.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])

    gr.Examples(
        label="Examples",
        examples=get_example(),
        inputs=[input_audio, src_prompt, tar_prompt, t_start, model_id, length, output_audio],
        outputs=[output_audio]
    )

    demo.queue()
    demo.launch(state_session_capacity=15)
