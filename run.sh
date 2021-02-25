#!/bin/bash

# Activate environment and install packages
conda init bash


conda install -c conda-forge -y librosa
conda install -c conda-forge -y tensorflow
## conda install -c conda-forge tensorflow-gpu
conda install -c conda-forge -y keras
git clone https://github.com/tapojyotipaul/DenseAE
cd DenseAE
python3 Dense_AE_Validation.py 
