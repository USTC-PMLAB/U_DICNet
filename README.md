# U-DICNet (Pytorch implementation)

U-DICNet estimates local complex deformation with high-order gradient from pairs of reference and deformed images of speckle pattern, as Digital Image Correlation (DIC) does. See paper [1] for details. 

If you find this implementation useful, please cite reference [1]. 

## Prerequisite

Install the following modules: 

```
pytorch >= 1.2
torchvision
tensorboardX 
imageio
argparse
numpy
pandas
cuda >=11.1
cudnn >= 7.5
```

## Training
1. Generate Speckle dataset
    dataset generation.m
    re*.bmp, tar*.bmp, u*.csv, v*.csv should be stored in the same folder
2. Specify the paths to:
    Train dataset, Test dataset
3. Execute the following commands
```
python Train.py --arch U_DICNet 
python Train.py --arch U_StrainNet_f
python Train.py --arch StrainNet_f
```

## Running inference

The images pairs should be in the same location, with the name pattern re*.ext  tar*.ext

```bash
python inference.py /path/to/input/images/  --arch U_DICNet  --pretrained /path/to/pretrained/model
python inference.py /path/to/input/images/  --arch U_StrainNet_f  --pretrained /path/to/pretrained/model
python inference.py /path/to/input/images/  --arch StrainNet_f  --pretrained /path/to/pretrained/model
```

## cellular deformation measurement

Execute the following commands in the U_DICNet directory (please also copy here the tar files if you use the pretrained models)

```bash
python inference.py ../sinusoidal_deformation/  --arch U_DICNet  --pretrained U_DICNet.pth.tar

```
The output of inference.py can be found in ../sinusoidal_deformation/


|Reference image   | ![](sinusoidal_deformation/re001.bmp)   |
|Target image      | ![](sinusoidal_deformation/tar001.bmp)  |
|:----------:|:---------------------------------------------:|
|Retrieved by U_DICNet  | ![](sinusoidal_deformation/U_DICNet.png)| or | ![](sinusoidal_deformation/U_DICNet_disp_x001.csv)|


## References 
[1] Lan S H, Su Y, Gao Z R, et al. Deep learning for complex displacement field measurement[J]. Science China Technological Sciences, 2022: 1-18.

https://link.springer.com/article/10.1007/s11431-022-2122-y

DOI
https://doi.org/10.1007/s11431-022-2122-y

## Acknowledgments

This code is based on the Pytorch implmentation of FlowNetS from [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
This code is based on the Pytorch implmentation of StrainNet from [StrainNetPytorch](https://github.com/DreamIP/StrainNet)
# U_DICNet
