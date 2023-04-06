# Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm (CVPR 2020)

<!-- Pytorch implementation of the OGNet for outliers detection as described in [Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.pdf).-->

[CVPR Presentation](https://www.youtube.com/watch?v=mAfAUwFlUpU) || [Paper](https://arxiv.org/abs/2004.07657) || [CVPR CVF Archive](http://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html) || [Supp material zip file](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zaheer_Old_Is_Gold_CVPR_2020_supplemental.zip) || [Ped2 Results Video](https://youtu.be/59Lqkkyy9bQ)

![img1](https://github.com/xaggi/OGNet/blob/master/imgs/OGNet_architect.png)

## Requirements

- Python2.7
- torch 1.2.0
- torchvision 0.4.0

Previously, the code was built using Python3.5, but as the version has reached its EOL, this code is verified on Python 2.7 now.

## Code execution

- Train.py is the entry point to the code.
- Place training and testing images under the directory 'data' by following the instructions provided in 'dataset.txt' file. 
- Set necessary options in opts.py for phase one and opts_fine_tune_discriminator.py for phase two.
- Execute Train.py

Previously, only test codes were provided for which test.py file was needed to run the evaluation. For that, the instructions can be found below. Note that, for the current version. test.py is not required as the code calls the test function every iteration from within to visualize the performance difference between the baseline and the OGNet.

- Download trained generator and discriminator models from [here](https://drive.google.com/drive/folders/1onNezvWJCfaKndvzOc3CXXnNidjosVvn?usp=sharing) and place inside the directory ./models/
- Download datasets [here](https://drive.google.com/drive/folders/1Cj28-1aV4AXtdm9j_CEs3UArLodc0GY4?usp=sharing) and place test images in the subdirectories of ./data/test/
  - Example:
    - All images from inlier class (\*.png) should be placed as ./data/test/0/*.png
    - Similarly, all images from outlier class (* \*.png) should be placed as ./data/test/1/* \*.png
- run test.py

## MNIST training and testing details
The [models provided](https://drive.google.com/drive/folders/1onNezvWJCfaKndvzOc3CXXnNidjosVvn?usp=sharing) are trained on the training set of '0' class in MNIST dataset. For evaluation, the test dataset [provided](https://drive.google.com/drive/folders/1Cj28-1aV4AXtdm9j_CEs3UArLodc0GY4?usp=sharing) contains all test images from class '0' as inliers, whereas 100 images each from all other classes as outliers.



## Updates

[17.6.2020] For the time being, test code (and some trained models) are being made available. Training code will be uploaded in some time.

[05.3.2021] A mockup training code is uploaded which can be used for training and evaluation of the model.



For any queries, please feel free to contact Zaigham through mzz . pieas @ gmail . com

If you find this code helpful, please cite our paper: 
 
```
  @inproceedings{zaheer2020old,
  title={Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm},
  author={Zaheer, Muhammad Zaigham and Lee, Jin-ha and Astrid, Marcella and Lee, Seung-Ik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14183--14193},
  year={2020}
  }
 ```

