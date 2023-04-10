import math
import warnings
from typing import Union, Optional, List

import torch
import torch.nn as nn
from kornia.geometry.linalg import compose_transformations, inverse_transformation

from ..odometry.dia import DIAOdometryProvider

# TODO I need an advanced pointcloud class
from ..odometry.icputils import downsample_pointclouds, downsample_rgbdimages
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .fusionutils import find_active_map_points, update_map_fusion


__all__ = ["VisualOdometryFrontend"]

class VisualOdometryFrontend(nn.Module):
    r"""_summary_

    Args:
        PointFusion (_type_): _description_
    """    

    def __init__(
        self, 
        *, 
        odom: str = "dia", 
        dist_th: Union[float, int] = 0.05, 
        angle_th: Union[float, int] = 20, 
        sigma: Union[float, int] = 0.6, 
        dsratio: int = 4, 
        numiters: int = 20, 
        damp: float = 1e-8, 
        dist_thresh: Union[float, int, None] = None, 
        lambda_max: Union[float, int] = 2, 
        lambda_geometric: Union[float, int] = 0.968, 
        B: Union[float, int] = 1, 
        B2: Union[float, int] = 1, 
        nu: Union[float, int] = 200, 
        device: Union[torch.device, str, None] = None
    ):
        super().__init__()
        if odom not in ["dia"]: # TODO gt, gradicp, icp
            msg = "odometry method ({}) not supported for PointFusion. ".format(odom)
            msg += "Currently supported odometry modules for PointFusion are: 'dia'."
            raise ValueError(msg)
        
        odomprov = None
        if odom == "dia":
            odomprov = DIAOdometryProvider(num_pyr_levels=dsratio, numiters=numiters)
        
        self.odom = odom
        self.odomprov = odomprov
        self.dsratio = dsratio
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = device
        
        if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
            raise TypeError(
                "Distance threshold must be of type float or int; but was of type {}.".format(
                    type(dist_th)
                )
            )
        if not (isinstance(angle_th, float) or isinstance(angle_th, int)):
            raise TypeError(
                "Angle threshold must be of type float or int; but was of type {}.".format(
                    type(angle_th)
                )
            )
        if dist_th < 0:
            warnings.warn(
                "Distance threshold ({}) should be non-negative.".format(dist_th)
            )
        if not ((0 <= angle_th) and (angle_th <= 90)):
            warnings.warn(
                "Angle threshold ({}) should be non-negative and <=90.".format(angle_th)
            )
        self.dist_th = dist_th
        rad_th = (angle_th * math.pi) / 180
        self.dot_th = torch.cos(rad_th) if torch.is_tensor(rad_th) else math.cos(rad_th)
        self.sigma = sigma
    
    def forward(self, frames: RGBDImages):
        pass

    def step(
        self, 
        pointclouds_list: List[Pointclouds], 
        live_frame: RGBDImages, 
        prev_frame: Optional[RGBDImages] = None, 
        inplace: bool = False
    ):
        if not isinstance(live_frame, RGBDImages):
            raise TypeError(
                "Expected live_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(live_frame)
                )
            )
        live_frame.poses, live_frame.all_poses, live_frame.init_T = self._localize(pointclouds_list, live_frame, prev_frame)
        pointclouds_list = self._map(pointclouds_list, live_frame, inplace)
        return pointclouds_list, live_frame.poses, live_frame.all_poses, live_frame.init_T

    def _localize(
        self, pointclouds_list: List[Pointclouds], live_frame: RGBDImages, prev_frame: RGBDImages
    ):
        if not isinstance(pointclouds_list, list):
            raise TypeError(
                "Expected pointclouds to be of type list. Got {0}.".format(
                    type(pointclouds_list)
                )
            )
        if not isinstance(live_frame, RGBDImages):
            raise TypeError(
                "Expected live_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(live_frame)
                )
            )
        if not isinstance(prev_frame, (RGBDImages, type(None))):
            raise TypeError(
                "Expected prev_frame to be of type gradslam.RGBDImages or None. Got {0}.".format(
                    type(prev_frame)
                )
            )
        if prev_frame is not None:
            if self.odom == "gt":
                warnings.warn(
                    "`prev_frame` is not used when using `odom='gt'` (should be None)"
                )
            elif not prev_frame.has_poses:
                raise ValueError("`prev_frame` should have poses, but did not.")
        if prev_frame is None and pointclouds_list[0].has_points and self.odom != "gt":
            msg = "`prev_frame` was None despite `{}` odometry method. Using `live_frame` poses.".format(
                self.odom
            )
            warnings.warn(msg)
        if prev_frame is None or self.odom == "gt":
            live_frame.segmented_RGBDs
            if not live_frame.has_poses:
                raise ValueError(
                    "`live_frame` must have poses when `prev_frame` is None or `odom='gt'`."
                )
            if prev_frame is not None:
                live_frame.init_T = compose_transformations(prev_frame.poses.squeeze(1), live_frame.poses.squeeze(1)).unsqueeze(1)
            return live_frame.poses, live_frame.all_poses, live_frame.init_T
        
        # Segment RGB and depth maps based on detected objects
        live_segments_ids = live_frame.segmented_RGBDs["ids"]
        prev_segments_ids = prev_frame.segmented_RGBDs["ids"]
        prev_segmented_RGBDs = prev_frame.segmented_RGBDs["rgbds"]

        all_poses = torch.zeros_like(live_frame._all_poses)

        for i in range(len(live_segments_ids)):
            id = live_segments_ids[i]
            idx = (prev_segments_ids == id).nonzero(as_tuple=True)[0]
            
            if idx.nelement() != 0:
                live_frame.segmented_RGBDs["rgbds"][i].poses = prev_segmented_RGBDs[idx].poses
                if id == 0:
                    transform = self.odomprov.provide(
                        live_frame.segmented_RGBDs["rgbds"][i], prev_segmented_RGBDs[idx]
                        )
                else:
                    transform = self.odomprov.provide(
                        live_frame,
                        prev_segmented_RGBDs[idx],
                    )

                live_frame.segmented_RGBDs["rgbds"][i].poses = compose_transformations(
                    prev_segmented_RGBDs[idx].poses.squeeze(1), 
                    inverse_transformation(transform.squeeze(1))
                ).unsqueeze(1)
                all_poses[:, :, id] = compose_transformations(
                    prev_segmented_RGBDs[idx].poses.squeeze(1), 
                    inverse_transformation(transform.squeeze(1))
                ).unsqueeze(1)
                live_frame.segmented_RGBDs["rgbds"][i].init_T = transform
            else: # Initialize new instance
                # Initialize new instance and take current frame as trajectory origin and as reference for following estimation
                all_poses[:, :, id] = torch.eye(4, device=self.device)
                live_frame.segmented_RGBDs["rgbds"][i].poses = live_frame.segmented_RGBDs["rgbds"][0].poses

        return live_frame.segmented_RGBDs["rgbds"][0].poses, all_poses, live_frame.segmented_RGBDs["rgbds"][0].init_T
    
        # TODO structure to store cameara poses and objects poses

    def _map(
        self, pointclouds_list: List[Pointclouds], live_frame: RGBDImages, inplace: bool = False
    ):
        if live_frame.channels_first:
            live_frame.to_channels_last_()

        live_obs = live_frame._O
        while live_obs > len(pointclouds_list):
            pointclouds_list.append(Pointclouds(device=self.device))
        for id in live_frame.segmented_RGBDs["ids"]:
            idx = (live_frame.segmented_RGBDs["ids"] == id).nonzero(as_tuple=True)[0]
            if self.has_enough_valid_pixels(live_frame.segmented_RGBDs["rgbds"][idx]):
                pointclouds_list[id] = update_map_fusion(
                    pointclouds_list[id], live_frame.segmented_RGBDs["rgbds"][idx], self.dist_th, self.dot_th, self.sigma, inplace
                )
        return pointclouds_list
    

    def initialize_object_traj():
        pass

    def has_enough_valid_pixels(self, rgbdimages: RGBDImages, th: int=30):
        r"""Check if the input rgbd frame contains an sufficient amount of valid pixels. If the number of valid depth value is below a threshold, return false. Otherwise, return true."""
        valid_depth = torch.sum(rgbdimages.depth_image > 0)
        return valid_depth > th