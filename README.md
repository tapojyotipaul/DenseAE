# Dense-AE-Validation
Dataset obtained from :https://github.com/MIMII-hitachi/mimii_baseline
## Steps to Run:
- Run run.sh directly (incase of tensorflow cpu)
- Uncomment line #9 and comment line #8 incase of tensorflow-gpu installation

## Steps to Run (Without run.sh):
- conda install -c conda-forge librosa
- conda install -c conda-forge tensorflow (For GPU use: conda install -c conda-forge tensorflow-gpu)
- conda install -c conda-forge keras
- git clone https://github.com/tapojyotipaul/DenseAE
- cd DenseAE
- python3 Dense_AE_Validation.py 