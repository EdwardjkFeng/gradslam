from typing import Union

import torch

from ..structures import RGBDImages
from .base import OdometryProvider
from .diautils import direct_image_align


__all__ = ["DIAOdometryProvider"]

class DIAOdometryProvider(OdometryProvider):
    r"""Direat method (direct image alignment) odometry provider using photometric error in a coarse-to-fine scheme. Computes the relative transformation between
    a pair of `gradslam.RGBDImages` objects by jointly minimizing the photometric error and depth error between them. Uses LM (Levenberg-Marquardt) solver.
    """

    def __init__(
        self,
        numPyrLevels: int = 3,
        numiters: int = 20,
        *params
    ):
        if not isinstance(numPyrLevels, int):
            raise TypeError(
                "numPyrLevels must be int. Got {0}.".format(type(numPyrLevels))
            )
        if not isinstance(numiters, int):
            raise TypeError(
                "numiters must be int. Got {0}.".format(type(numiters))
            )
        self.numPyrLevels = numPyrLevels
        self.numiters = numiters
    
    def provide(
        self,
        pre_frame: RGBDImages,
        cur_frame: RGBDImages,
    ) -> torch.Tensor:
        r"""Uses direct image alignment to compute the relative homogenous transformation that, when applied to `cur_frame`,
        would cause the pixels to align with pixels of `pre_frame`.

        Args:
            pre_frame (gradslam.RGBDImage): Object containing batch of previous rgbd images of batch size
                :math:`(B)`
            cur_frame (gradslam.RGBDImage): Object containing batch of current rgbd images of batch size
                :math:`(B)`

        Returns:
            torch.Tensor: The relative transformation that would align `cur_frame` with `pre_frame`

        Shape:
            - Output: :math:`(B, 1, 4, 4)`

        """
        if not isinstance(pre_frame, RGBDImages):
            raise TypeError(
                "Expected pre_frame to be of type gradslam.RGBDImages. Got {0}.".format(type(pre_frame))
            )
        if not isinstance(cur_frame, RGBDImages):
            raise TypeError(
                "Expected cur_frame to be of type gradslam.RGBDImages. Got {0}.".format(type(cur_frame))
            )
        
        device = pre_frame.device
        initial_transform = torch.eye(4, device=device)

        transforms = []
        for b in range(len(pre_frame)): # iterate over the batches
            transform = direct_image_align(
                cur_frame[b],
                pre_frame[b],
                initial_transform,
                numPyrLevels=self.numPyrLevels,
                numiters=self.numiters,
            )
        
            transforms.append(transform)
        
        return torch.stack(transforms).unsqueeze(1)