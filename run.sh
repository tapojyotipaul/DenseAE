#!/bin/bash

# Activate environment and install packages
conda init bash


conda install -c conda-forge librosa
conda install -c conda-forge tensorflow
## conda install -c conda-forge tensorflow-gpu
conda install -c conda-forge keras
git clone https://github.com/tapojyotipaul/DenseAE
cd DenseAE
python3 Dense_AE_Validation.py 
