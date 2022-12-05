# Import gradslam related modules
import gradslam as gs
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import TUM
from gradslam.slam import PointFusion


import matplotlib as plt
import numpy as np
import os 
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # Down TUM dataset
    data_path = '../data/'
    if not os.path.isdir(data_path + 'TUM'):
        os.mkdir(data_path + 'TUM')
    if not os.path.isdir(data_path + 'TUM/rgbd_dataset_freiburg1_desk'):
        print('No dataset found in ', data_path)
        # print('Downloading TUM/rgbd_dataset_freiburg1_desk dataset ...')
        # os.mkdir(data_path + 'TUM/rgbd_dataset_freiburg1_desk')
        # !wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
        exit()

    tum_path = data_path + 'TUM/'

    # Load data
    dataset = TUM(tum_path, seqlen=4, height=480, width=640)
    loader = DataLoader(dataset=dataset, batch_size=2)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))


    # Create RGB-D image objects 
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
    rgbdimages.plotly(0).update_layout(autosize=False, height=600, width = 800).show()
