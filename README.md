# VGIS10_GAN
Repository for 10th semester master thesis at VGIS on Aalborg University.

[OpenGANscripts](./OpenGANscripts/) are created from the Jupyter NoteBooks of [OpenGAN](https://github.com/aimerykong/OpenGAN).
[config_HRNet](./OpenGANscripts/config_HRNet/) folder is from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

[AnomalyDetection](./AnomalyDetection/) contains files from [MNADrc](https://github.com/alchemi5t/MNADrc)

## Install procedure:
Made on WSL2 installed with conda.

Clone this repo into your prefered folder:
`https://github.com/TheCanMoun10/VGIS10_GAN`

Install the required dependencies use the environment.yml file available:
`conda env create -f environment.yml`


# DATASETS FOR THE GAN CAN BE FOUND HERE.
* CUHK Avenue [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into ``datasets`` folder, like ``./datasets/avenue/``

## Other datasets that can be used:
* ShanghaiTech [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
* USCD Ped2 [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)