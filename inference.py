import argparse
import torch
import torch.backends.cudnn as cudnn
import models
import time
import numpy as np
import os
import glob
import cv2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='U-DICNet inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='U_DICNet', choices=['U_DICNet', 'U_StrainNet_f', 'StrainNet_f'],
                    help='network archtecture')
parser.add_argument('--img_dir', metavar='DIR',
                    help='path to images folder, image names must match \'re*.[ext]\' and \'tar*.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--save_path', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in image folder')

parser.add_argument("--img-exts", metavar='EXT', default=['tif', 'png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# device = torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    # img_dir=input('Please input the image path:')
    # pre_train=input('Please input the pre_train path:')
    # save_path=input('Please input the save path:')

    img_dir = args.img_dir
    pre_train = args.pretrained

    if args.save_path is None:
        save_path = img_dir
    else:
        save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # list to store the image name of the processing image pair.
    img_pairs = []

    # find the image pair in the image folder
    for ext in args.img_exts:
        for file in sorted(glob.glob(os.path.join(img_dir, 're*.{}'.format(ext)))):
            re_img_name = os.path.basename(file)
            tar_img_name = 'tar' + re_img_name[2:]
            tar_img_dir = os.path.join(img_dir, tar_img_name)
            if os.path.isfile(tar_img_dir):
                img_pairs.append([file, tar_img_dir])

    # load the pretrained network model
    network_data = torch.load(pre_train)
    model = models.__dict__[args.arch](network_data).to(device)
    model.eval()
    cudnn.benchmark = True
    img_num = 0
    for (img1_file, img2_file) in img_pairs:
        # read the images
        re_img = np.array(cv2.imdecode(np.fromfile(img1_file, dtype=np.uint8), 0))
        tar_img = np.array(cv2.imdecode(np.fromfile(img2_file, dtype=np.uint8), 0))

        # RDB to gray
        if len(re_img.shape) > 2:
            re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)

        # normalization
        re_img = (re_img - np.mean(re_img)) / np.max(np.abs(re_img - np.mean(re_img)))
        tar_img = (tar_img - np.mean(tar_img)) / np.max(np.abs(tar_img - np.mean(tar_img)))

        # compute the displacement
        if args.arch != 'StrainNet_f':
            disp_x = compute_displacement(re_img, tar_img, model)
            disp_y = compute_displacement(re_img.T, tar_img.T, model).T
        else:
            disp = compute_displacement(re_img, tar_img, model)
            disp_x = disp[0, :, :]
            disp_y = disp[0, :, :]

        img_num = img_num + 1

        # save the result
        np.savetxt(save_path + '/U_DICNet_dispx_' + img2_file[-7:-4] + '.csv', disp_x[:, :], delimiter=',')
        np.savetxt(save_path + '/U_DICNet_dispy_' + img2_file[-7:-4] + '.csv', disp_y[:, :], delimiter=',')


def compute_displacement(re_img, tar_img, model):
    """
    Args:
        re_img : the refrence image
        tar_img : the target image
        model : the network model

    Returns:
        disp : the predicted displacement field
               U_DICNet and U_StrainNet_f : only the displacement field in the X direction was predicted
               StrainNet_f : both the displacement field in the X and Y directions were predicted
    """

    global args

    # disp_x = np.zeros((re_img.shape[0], re_img.shape[1]))
    re_img = torch.from_numpy(re_img).float().to(device)
    tar_img = torch.from_numpy(tar_img).float().to(device)
    re_img = re_img[np.newaxis, ...]
    tar_img = tar_img[np.newaxis, ...]

    # different network architecture have different input
    # StrainNet_f: |re_img tar_img|re_img tar_img|re_img tar_img|
    # U_StrainNet_f and U_DICNet: |re_img tar_img|
    if args.arch != 'StrainNet_f':
        input_img = torch.cat([re_img, tar_img], 0)
    else:
        input_img = torch.cat([re_img, tar_img, re_img, tar_img,re_img, tar_img], 0)
    input_img = input_img[np.newaxis, ...]
    start = time.time()
    output = model(input_img)

    # record the caculation time
    time_used = (time.time() - start)
    # print(time_used)

    output_to_write = output.data.cpu()

    output_to_write = output_to_write.numpy()
    disp = output_to_write.squeeze()

    return disp


if __name__ == '__main__':
    main()
