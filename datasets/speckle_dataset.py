import os.path
import os
from .image_dataset import image_dataset
import glob


def make_dataset(file_path, network_arch):
    """
    Args:
        file_path: the path of the image data
        network_arch : the network archtecture

    Returns:
        re_img_list: the list of filename of the reference image
        tar_img_list: the list of filename of the target image
        displacement_list: the list of filename of the real displacment field
    """

    re_img_list = []
    tar_img_list = []
    displacement_list = []

    # find the reference image, the target image, and the displacement field in the file path
    # the reference image: *re*.bmp
    # the target image: *tar*.bmp
    # the displacement field: *u*.csv   *v*.csv
    for re_img in sorted(glob.glob(os.path.join(file_path, '*re*.bmp'))):
        re_img = os.path.basename(re_img)  # 返回路径中的最后一个文件名
        # root_filename = re_img[2:]
        img1 = re_img
        if not (os.path.isfile(os.path.join(file_path, img1))):
            continue
        re_img_list.append(img1)
    for tar_img in sorted(glob.glob(os.path.join(file_path, '*tar*.bmp'))):
        tar_img = os.path.basename(tar_img)  # 返回路径中的最后一个文件名
        # root_filename2 = tar_img[3:]
        img2 = tar_img  # 'tar' + root_filename2
        if not (os.path.isfile(os.path.join(file_path, img2))):
            continue
        tar_img_list.append(img2)
    for dis_csv in sorted(glob.glob(os.path.join(file_path, '*u*.csv'))):
        dis_csv = os.path.basename(dis_csv)  # 返回路径中的最后一个文件名
        dis_name_u = dis_csv  # 'u' + root_filename3
        dis_name_v = dis_csv.replace('u', 'v')  # 'v' + root_filename3
        if not (os.path.isfile(os.path.join(file_path, dis_name_u))):
            continue
        """
        different network architecture have different outputs
        StrainNet_f: both the displacement in X and Y direction were used
        U_StrainNet_f and U_DICNet: only the displacement in the X direction was used
        """
        if (network_arch == 'StrainNet_f') & os.path.isfile(os.path.join(file_path, dis_name_v)):
            displacement_list.append([dis_name_u, dis_name_v])
        else:
            displacement_list.append(dis_name_u)
    return re_img_list, tar_img_list, displacement_list


def speckle_dataset(train_root, test_root, network_arch):

    train_re_img_list, train_tar_img_list, train_displacement_list = make_dataset(train_root, network_arch)
    test_re_img_list, test_tar_img_list, test_displacement_list = make_dataset(test_root, network_arch)

    train_dataset = image_dataset(train_root, train_re_img_list, train_tar_img_list, train_displacement_list, network_arch)
    test_dataset = image_dataset(test_root, test_re_img_list, test_tar_img_list, test_displacement_list, network_arch)
    return train_dataset, test_dataset
