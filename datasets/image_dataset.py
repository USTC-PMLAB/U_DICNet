import torch
from imageio import imread
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd


def default_loader(root, re_img_name, tar_img_name, displacement, network_arch):
    """
    Args:
        root: the path of the image data
        re_img_name : the filename of the reference image
        tar_img_name : the filename of the reference image
        displacement : the filename of the displacement field
        network_arch : the network archtecture

    Returns:
        re_img: the reference image
        tar_img: the target image
        output_dis: the real displacment field
    """

    # read the image data
    re_img = imread(os.path.join(root, re_img_name))
    tar_img = imread(os.path.join(root, tar_img_name))

    # normalization
    re_img = (re_img - np.mean(re_img)) / np.max(np.abs(re_img - np.mean(re_img)))
    tar_img = (tar_img - np.mean(tar_img)) / np.max(np.abs(tar_img - np.mean(tar_img)))

    # different network architecture have different outputs
    # StrainNet_f: both the displacement in X and Y direction were used
    # U_StrainNet_f and U_DICNet: only the displacement in the X direction was used
    if network_arch == 'StrainNet_f':
        output_dis_u = np.array(pd.read_csv(os.path.join(root, displacement[0]), header=None))
        output_dis_v = np.array(pd.read_csv(os.path.join(root, displacement[1]), header=None))
        output_dis = np.zeros([2, output_dis_u.shape[0], output_dis_u.shape[1]])
        output_dis[0, :, :] = output_dis_u
        output_dis[1, :, :] = output_dis_v
    else:
        output_dis = np.array(pd.read_csv(os.path.join(root, displacement), header=None))

    return re_img, tar_img, output_dis


class image_dataset(Dataset):
    """
    dataset
    Args:
        root: the path of the image data
        re_img_list : the filename of the reference image
        tar_img_list : the filename of the reference image
        displacement_list : the filename of the displacement field
        network_arch : the network archtecture
        loader : data loader (the defined default_loader)
    """
    def __init__(self, root, re_img_list, tar_img_list, displacement_list, network_arch='U_DICNet',
                 loader=default_loader):
        self.root = root
        self.re_img_list = re_img_list
        self.tar_img_list = tar_img_list
        self.displacement_list = displacement_list
        self.network_arch = network_arch
        self.loader = loader

    def __getitem__(self, index):
        # load the image the displacement field
        input1, input2, target = self.loader(self.root, self.re_img_list[index], self.tar_img_list[index],
                                            self.displacement_list[index], self.network_arch)

        # transform to tensor
        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        target = torch.from_numpy(target).float()

        input1 = input1[np.newaxis, ...]
        input2 = input2[np.newaxis, ...]

        # different network architecture have different input
        # StrainNet_f: |re_img tar_img|re_img tar_img|re_img tar_img|
        # U_StrainNet_f and U_DICNet: |re_img tar_img|
        if self.network_arch == 'StrainNet_f':
            input_img = torch.cat((input1, input2), 0)
            input_img = torch.cat((input_img, input_img, input_img), 0)
        else:
            input_img = torch.cat((input1, input2), 0)
            target = target[np.newaxis, ...]
        # return the input images and target displacement
        return input_img, target

    def __len__(self):
        return len(self.re_img_list)
