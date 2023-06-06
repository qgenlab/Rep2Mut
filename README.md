# Rep2Mut
This repository includes the implementation of Rep2Mut: “Accurate Prediction of Transcriptional Activity of Single Missense Variants in HIV Tat with Deep Learning” and Rep2Mut-V2:” Accurate prediction of functional effect of single missense variants with deep learning”. Please cite our papers if you use the models or codes. In this package, we provides resources including: source codes of the Rep2Mut model, source codes of the Rep2Mut-V2 model, and usage examples. 

## Environment setup
We recommend you to build a python virtual environment with Anaconda. Also, if you want to train the model, make sure that you have at least one NVIDIA GPU. We have trained the model on NVIDIA a100 sxm4 with 80 GB graphic memory. If you use GPU with other specifications and memory sizes, consider adjusting your batch size accordingly.

### Create a new virtual environment
`conda env create -f environment.yml`

### Activate the created environment
`conda activate py39Rep2Mut`

## 

## Reference

Derbel, Houssemeddine, et al. "Accurate prediction of transcriptional activity of single missense variants in HIV Tat with deep learning." International Journal of Molecular Sciences 24.7 (2023): 6138.

Derbel, Houssemeddine, et al. "Accurate prediction of functional effect of single missense variants with deep learning". (Accepted by ICIBM 2023)

