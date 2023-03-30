# VGIS10_GAN
Repository for 10th semester master thesis at VGIS on Aalborg University.

[OpenGANscripts](./OpenGANscripts/) are created from the Jupyter NoteBooks of [OpenGAN](https://github.com/aimerykong/OpenGAN).
[config_HRNet](./OpenGANscripts/config_HRNet/) folder is from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

[AnomalyDetection](./AnomalyDetection/) contains files from [MNADrc](https://github.com/alchemi5t/MNADrc)

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

## Training
* ~~The training and testing codes are based on prediction method~~
* Now you can implemnet the codes based on both prediction and reconstruction methods.
* The codes are basically based on the prediction method, and you can easily implement this as
```bash
https://github.com/TheCanMoun10/VGIS10_GAN
cd projects/MNAD/code
python Train.py # for training
```
* You can freely define parameters with your own settings like
```bash
python Train.py --gpus 1 --dataset_path 'your_dataset_directory' --dataset_type avenue --exp_dir 'your_log_directory'
```

## Evaluation
* Test your own model
* Check your dataset_type (ped2, avenue or shanghai)
```bash
python Evaluate_recons_wo_mem.py --t_length 2 --alpha 0.7 --th 0.015 --dataset_type avenue --model_dir ./path/to/model.pth
```

# DATASETS FOR THE GAN CAN BE FOUND HERE.
* CUHK Avenue [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".

Download the datasets into ``datasets`` folder, like ``./datasets/avenue/``

## Other datasets that can be used:
* ShanghaiTech [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
* USCD Ped2 [dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)