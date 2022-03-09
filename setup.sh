#!/bin/bash
conda env update -f environment.yml
conda activate zeroshot
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
conda deactivate
