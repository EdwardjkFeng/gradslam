import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
import imageio.v3 as imageio
import torch
import matplotlib.pyplot as plt

SHOW_DEPTH = False
SHOW_RGB = True
SHOW_MASK = False

data_dir = '/home/jingkun/Dataset/CoFusion/'
sequence = 'room4-full'
# data_dir = '/home/jingkun/Dataset/TUM/'
# sequence = 'rgbd_dataset_freiburg1_desk'

color_dir = os.path.join(data_dir, sequence + '/', 'colour/')
depth_dir = os.path.join(data_dir, sequence + '/', 'depth_noise/')
mask_dir = os.path.join(data_dir, sequence + '/', 'mask_colour/')
# color_dir = os.path.join(data_dir, sequence + '/', 'rgb/')
# depth_dir = os.path.join(data_dir, sequence + '/', 'depth/')

num_rgb_frames = len([f for f in os.listdir(color_dir) if os.path.isfile(color_dir + f)])
num_depth_frames = len([f for f in os.listdir(depth_dir) if os.path.isfile(depth_dir + f)])

print('Sequence {} contains {} RGB images and {} depth images.'.format(sequence, num_rgb_frames, num_depth_frames))

seq_len = 850
for i in range(1, seq_len+1):
    color = cv2.imread(color_dir + "Color{0:04d}.png".format(i), cv2.IMREAD_UNCHANGED)
    depth = np.asarray(imageio.imread(depth_dir + "Depth{0:04d}.exr".format(i)), dtype=np.float32)
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    mask = np.asarray(imageio.imread(mask_dir + "Mask{0:04d}.png".format(i)), dtype=np.int8)

    # Filter out moving objects 
    mask_filter = np.sum(mask, axis=2) != 0
    color[mask_filter, :] = [0, 0, 0]


    # # Visualize with plt
    # print(depth.shape)
    # new_depth = depth[:, :, 0]
    # fig = plt.figure()
    # plt.imshow(new_depth)
    # plt.colorbar()
    # plt.show()

    # # Visualise with opencv
    # new_depth = cv2.normalize(new_depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imshow('depth', new_depth)
    # cv2.waitKey(0)

    # if len(depth.shape) ==3 and depth.shape[2] == 3:
    #     row, col, channel = depth.shape
    #     new_depth = np.empty((row, col), dtype=np.float32).reshape(-1)
    #     depth_idx = 0
    #     for i in range(row):
    #         for j in range(col):
    #             new_depth[depth_idx] = depth[i, j, 0]
    #             depth_idx += 1
    #     new_depth = np.reshape(new_depth, (row, col))

    # Visualize rgb image
    if SHOW_RGB:
        cv2.imshow('Color', color)
        cv2.waitKey(10)

    # Visualize depth image
    if SHOW_DEPTH:
        depth = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('Depth', depth)
        cv2.waitKey(10)

    # Visualize mask
    if SHOW_MASK:
        cv2.imshow('Mask', mask)
        cv2.waitKey(10)

