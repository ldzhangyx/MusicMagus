
<div align="center">

# 🪄MusicMagus

[![python](https://img.shields.io/badge/-Python_3.1+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2402.06178-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/IJCAI-2024-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the official repository for the paper "MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models".

If there is any problem related to the code running, please open an issue and I will help you as mush as I can.

## Demo page

https://bit.ly/musicmagus-demo

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/ldzhangyx/MusicMagus/
cd MusicMagus

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11.7
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/ldzhangyx/MusicMagus/
cd MusicMagus

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## Configuring LP-MusicCaps

Please download `transfer.pth` from the website below and place it to `lpmc/music_captioning/exp/transfer/lp_music_caps/` folder.

```
https://huggingface.co/seungheondoh/lp-music-caps/blob/main/transfer.pth
```

## How to run

1. Set the `openai.key` value in `audioldm2/embedding_calculator.py`.

2. Directly run `inference.ipynb`.

## Citation

```
@misc{zhang2024musicmagus,
      title={MusicMagus: Zero-Shot Text-to-Music Editing via Diffusion Models}, 
      author={Yixiao Zhang and Yukara Ikemiya and Gus Xia and Naoki Murata and Marco A. Martínez-Ramírez and Wei-Hsiang Liao and Yuki Mitsufuji and Simon Dixon},
      year={2024},
      eprint={2402.06178},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
