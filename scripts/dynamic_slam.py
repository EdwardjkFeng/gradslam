import gradslam as gs
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import Cofusion, ICL, TUM
from gradslam.slam import VisualOdometryFrontend

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch.utils.data import DataLoader

import open3d as o3d
import copy
import time
import gc


def load_data(
    n_frames=10, 
    data_set='CoFusion', 
    data_path='/home/jingkun/Dataset/', 
    dilation=3,
    start=0,
    height=240,
    width=320,
    load_masks = False
):
    match data_set:
        case 'CoFusion':
            cofusion_path = data_path + 'CoFusion/'

            sequences = ("room4-full",) # 850 frames

            # Load data
            dataset = Cofusion(basedir=cofusion_path, sequences=sequences, seqlen=n_frames, dilation=dilation, start=start, height=height, width=width, channels_first=False, return_object_mask=load_masks)
        case 'ICL':
            icl_path = data_path + 'ICL/' # associated 880 frames

            # load dataset
            dataset = ICL(icl_path, trajectories=("living_room_traj1_frei_png",) ,seqlen=n_frames, dilation=dilation, start=start, height=height, width=width, channels_first=False)

        case 'TUM':
            tum_path = data_path + 'TUM/'
            sequences = ("rgbd_dataset_freiburg1_xyz",) # associated 792 frames

            # Load data
            dataset = TUM(basedir= tum_path, sequences=sequences, seqlen=n_frames, dilation=dilation, start=start, height=height, width=width)

    loader = DataLoader(dataset=dataset, batch_size=1)
    if load_masks:
        colors, depths, intrinsics, poses, *_, names, masks, labels = next(iter(loader))
    else:
        colors, depths, intrinsics, poses, *_, names = next(iter(loader))

    # Debug info
    print(f"colors shape: {colors.shape}")  # torch.Size([2, 8, 240, 320, 3])
    print(f"depths shape: {depths.shape}")  # torch.Size([2, 8, 240, 320, 1])
    print(f"intrinsics shape: {intrinsics.shape}")  # torch.Size([2, 1, 4, 4])
    print(f"poses shape: {poses.shape}")  # torch.Size([2, 8, 4, 4])
    print(f"masks shape: {masks.shape}") if load_masks else None# torch.Size([1, 200, 480, 640, 3])
    print(f"labels shape: {labels.shape}") if load_masks else None
    print('---')

    # Create RGB-D image objects 
    rgbdimages = RGBDImages(colors.requires_grad_(False),
                            depths.requires_grad_(False), 
                            intrinsics.requires_grad_(False),
                            poses.requires_grad_(False),
                            channels_first=False,
                            object_mask=masks,
                            object_label=labels,
                            )
    
    # Clean up cache
    del loader, colors, depths, poses
    gc.collect()
    torch.cuda.empty_cache()

    return rgbdimages, intrinsics

def initialize_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1000, width=1000)
    return vis



if __name__ == "__main__":
    NUM_FRAMES = 50
    START_FRAME = 500
    NUM_ITERATION = 30
    NUM_PYRAMID = 3
    ODOM = "dia"

    input_frames, intrinsics = load_data(n_frames=NUM_FRAMES, start=START_FRAME, load_masks=True)
    # TODO implement a different visualizer
    # input_frames.plotly(0).show()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    slam = VisualOdometryFrontend(odom=ODOM, numiters=NUM_ITERATION, dsratio=NUM_PYRAMID, device=device)
    pcds = Pointclouds(device=device)
    pcds_list = [pcds]
    batch_size, seq_len = input_frames.shape[:2]
    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).expand(batch_size, -1, -1, -1)
    prev_frame = None
    poses = []

    vis = initialize_visualizer()
    intermediate_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(intermediate_pcd)

    R_y_180 = np.eye(4, dtype=float)
    R_y_180[1, 1] = R_y_180[2, 2] = -1.0

    # Initialize coordinate system and fustum in open3D
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) #.transform(R_z_180)
    INTRINSIC = intrinsics.squeeze().numpy()[:3, :3]
    INT_POSE = np.eye(4)
    fustum_int = o3d.geometry.LineSet.create_camera_visualization(320, 240, INTRINSIC, INT_POSE, 0.07)
    camera_pose = None
    fustum = None

    s = 0
    keep_running = True
    while keep_running:
        if s < seq_len:
            vis.remove_geometry(intermediate_pcd)
            # if camera_pose is not None:
            #     vis.remove_geometry(camera_pose)
            # if fustum is not None:
            #     vis.remove_geometry(fustum) 
            
            live_frame = input_frames[:, s].to(device)
            if s == 0 and live_frame.poses is None:
                live_frame.poses = initial_poses

            pcds_list, live_frame.poses, live_frame.all_poses, live_frame.init_T = slam.step(pcds_list, live_frame, prev_frame, inplace=False)

            for id in live_frame.segmented_RGBDs["ids"]:
                intermediate_pcd = pcds_list[id].open3d(0, max_num_points=100*10000)
                intermediate_pcd.transform(R_y_180)
                vis.add_geometry(intermediate_pcd)

            pose = live_frame.poses.cpu().detach().squeeze().numpy()
            camera_pose = copy.deepcopy(world_frame).transform(pose).transform(R_y_180)

            fustum = copy.deepcopy(fustum_int).transform(pose).transform(R_y_180)
            vis.add_geometry(camera_pose)
            vis.add_geometry(fustum)

            poses.append(pose)

            prev_frame = live_frame if ODOM != "gt" else None

        s += 1
        keep_running = vis.poll_events()
        vis.update_renderer()

vis.destroy_window()