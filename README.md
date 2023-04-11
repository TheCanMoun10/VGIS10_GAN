# VGIS10_GAN
Repository for 10th semester master thesis at VGIS on Aalborg University.

[OpenGANscripts](./OpenGANscripts/) are created from the Jupyter NoteBooks of [OpenGAN](https://github.com/aimerykong/OpenGAN).
[config_HRNet](./OpenGANscripts/config_HRNet/) folder is from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

[AnomalyDetection](./AnomalyDetection/) contains files from [MNADrc](https://github.com/alchemi5t/MNADrc).

## Install procedure:
Made on WSL2 installed with conda.

Clone this repo into your prefered folder:
```bash
https://github.com/TheCanMoun10/VGIS10_GAN
```

Install the required dependencies use the environment.yml file available:
```bash
conda env create -f environment.yml
```

## Dependencies
* Python 3.8
* PyTorch 2.0.0
* Cuda 11.8
* Numpy
* Sklearn

## OGNet Training and testing:
Before training make sure to create the necessary folders:
```bash
mkdir models
mkdir results
```

```bash
https://github.com/TheCanMoun10/VGIS10_GAN
cd OGet
python train.py --data_path ./data/avenue_full/training --nc 3 --normal_class frames --epoch 26 --batch_size 32 --n_threads 0 --image_grids_numbers 30 --wandb
```

## Datasets.
* CUHK Avenue [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into [datasets](./AnomalyDetection/datasets/) folder, like ``./datasets/avenue/``
For OGNet the testing folder should be divided into two subfolders ``frames`` and ``frames2``

### Other datasets that can be used:
* ShanghaiTech [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
* USCD Ped2 [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
