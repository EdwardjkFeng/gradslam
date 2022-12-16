# Import gradslam related modules
import gradslam as gs
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import TUM, ICL
from gradslam.slam import PointFusion


import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
from torch.utils.data import DataLoader

import open3d as o3d
import gc
import time


if __name__ == '__main__':

    torch.cuda.empty_cache()

    data_path = '/home/jingkun/Dataset/'
    tum_path = data_path + 'TUM/'
    sequences = ('rgbd_dataset_freiburg1_xyz',)

    # # Load data
    # path = data_path + tum_path + sequences[0] + '/'
    # num_rgb_frames = len([name for name in os.listdir(path + 'rgb') if os.path.isfile(path + 'rgb/' + name)])
    # num_depth_frames = len([name for name in os.listdir(path + 'depth') if os.path.isfile(path + 'depth/' + name)])
    # # seq_len = min(num_rgb_frames, num_depth_frames)
    seq_len = 100

    dataset = TUM(tum_path, sequences, seqlen=seq_len, dilation=2, height=480, width=640)
    loader = DataLoader(dataset=dataset, batch_size=1)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))


    # Create RGB-D image objects (without storing gradients)
    rgbdimages = RGBDImages(colors.requires_grad_(False),
                            depths.requires_grad_(False), 
                            intrinsics.requires_grad_(False),
                            # poses.requires_grad_(False),
                            )

    device = torch.device("cuda") if torch.cuda.is_available() \
                                  else torch.device("cpu")

    rgbdimages.to(device=device)
    slam = PointFusion(odom='gradicp', device=device).requires_grad_(False)
    global_map = Pointclouds(device=device)
    batch_size, seq_len = rgbdimages.shape[:2]
    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    prev_frame = None

    start = time.time()
    # local_frames_counter = 0

    list_t_frame = []
    curr_local_map = Pointclouds(device=device)
    prev_local_map = Pointclouds(device=device)

    for s in range(seq_len):
        start = time.time()

        live_frame = rgbdimages[:, s].to(device)
        # print(live_frame.poses)
        
        if s == 0 and live_frame.poses is None:
            live_frame.poses = initial_poses
        curr_local_map, live_frame.poses = slam.step(prev_local_map, live_frame, prev_frame, inplace=True)
        # print(live_frame.poses)
        prev_frame = live_frame if slam.odom != 'gt' else None

        list_t_frame.append((time.time() - start))
        print("Frame: %d, Time: %.3f" % (s, list_t_frame[-1]))

        # global_map.append_points(curr_local_map)
        prev_local_map = curr_local_map

        del curr_local_map
        gc.collect()
        torch.cuda.empty_cache() 
    # o3d.visualization.draw_geometries([global_map.open3d(0, max_num_points=None)])


    # pcds = Pointclouds(device=device)

    # for s in range(seq_len):
    #     start = time.time()
    #     live_frame = rgbdimages[:, s].to(device)
    #     if s == 0 and live_frame.poses is None:
    #         live_frame.poses = initial_poses
    #     pcds, live_frame.poses = slam.step(pcds, live_frame, prev_frame, inplace=True)
    #     prev_frame = live_frame if slam.odom != 'gt' else None

    #     list_t_frame.append((time.time() - start))
    #     print("Frame: %d, Time: %.3f" % (s, list_t_frame[-1]))

    # o3d.visualization.draw_geometries([pcds.open3d(0, max_num_points=seq_len*100000)])



    # Plot graph of consumed time per frame
    plt.plot(list_t_frame)
    plt.ylabel('Consumed time per frame [ms]')
    plt.show()
