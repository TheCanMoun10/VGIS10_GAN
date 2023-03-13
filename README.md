# VGIS10_GAN
Repository for 10th semester master thesis at VGIS on Aalborg University.

Utils folder and Python file created from the Jupyter NoteBooks of [OpenGAN](https://github.com/aimerykong/OpenGAN).

config_HRNet folder is from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

## Install procedure:
Made on WSL2.

Clone this repo into you prefered folder:
`https://github.com/TheCanMoun10/VGIS10_GAN`


Make sure your your Ubuntu distribution is up to date (this repo uses 20.04):
`sudo apt update`

`sudo apt upgrade`

(Optional: `sudo apt autoremove`, to remove unnecessary packages)

Make sure to install the gcc and g++ compilers:

`sudo apt install gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++`

An anaconda environment file is provided to set up the dependencies for this GAN-network, to install use:

`conda env create -f environment.yml`
`conda activate VGIS10GAN`

Next install pytorch in the anaconda environment:

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

#TODO: Make environment file with pytorch in it.

## TinyImageNet
To download the TinyImageNet use the following commands in the terminal:
	
  `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`
  
  `unzip -qq tiny-imagenet-200.zip`
