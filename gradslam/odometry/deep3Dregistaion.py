from typing import Optional, Union

import torch

from ..structures.pointclouds import Pointclouds
from .base import OdometryProvider
from .deep3Dregistrationutils import deep_3d_registraiton

__all__ = ["Deep3DRegistrationProvider"]

class Deep3DRegistrationProvider(OdometryProvider):
    r"""ICP odometry provider using a combination of photometric error and a 
    point-to-plane error metric. Computes the relative transformation between
    a pair of 'gradslam.Pointclouds' objects using ICP (Iterative Closest Point).
    Uses LM (Levenberg-marquardt) solver.
    """

    def __init__(
        self
    ):
        r"""Initializes internal ColorICPOdometryProvider state.
        
        Args:
            numiters (int): Number of iterations to run the optimization for. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Default: 1e-8
            dist_thresh (float or int or None): Distance threshold for removing 'src_pc' points distant from 'tgt_pc'.
                Default: None
        """

    def provide(
        self, 
        maps_pointclouds: Pointclouds,
        frames_pointclouds: Pointclouds,
    ) -> torch.Tensor:
        r"""Uses color ICP to compute the relative homogenous transformation that, when applied to `frames_pointclouds`,
        would cause the points to align with points of `maps_pointclouds`.

        Args:
            maps_pointclouds (gradslam.Pointclouds): Object containing batch
            of map pointclouds of batch siue 
                :math:'(B)´
            frames_pointclouds (gradslam.Pointclouds): Object containinit batch or live frame pointclouds of batch siye
                :math:´(B)´
        
        Returns: 
            torch.Tensor: The relative transformation that would align ´maps_pointclouds´ with ´frame_pointclouds´
        
        Shapes:
            - Output: :math:'(B, 1, 4, 4)´
        """
        if not isinstance(maps_pointclouds, Pointclouds):
            raise TypeError(
                "Expected maps_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(maps_pointclouds)
                )
            )
        if not isinstance(frames_pointclouds, Pointclouds):
            raise TypeError(
                "Expected frames_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(frames_pointclouds)
                )
            )
        if maps_pointclouds.normals_list is None:
            raise ValueError(
                "maps_pointclouds missing normals. Map normals must be provided if using ColorICPOdometryProvider"
            )
        if len(maps_pointclouds) != len(frames_pointclouds):
            raise ValueError(
                "Batch size of maps_pointclouds and frames_pointclouds should be equal ({0} != {1})".format(
                    len(maps_pointclouds), len(frames_pointclouds)
                )
            )
        
        device = maps_pointclouds.device
        initial_transform = torch.eye(4, device=device)
        
        transforms = []
        for b in range(len(maps_pointclouds)):
            # TODO: use color icp
            transform = deep_3d_registraiton(
                frames_pointclouds.points_list[b].unsqueeze(0),
                frames_pointclouds.normals_list[b].unsqueeze(0),
                maps_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.normals_list[b].unsqueeze(0),
                initial_transform,
            )

            transforms.append(transform)
        
        return torch.stack(transforms).unsqueeze(1)
