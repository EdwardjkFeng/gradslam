from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from torchmetrics.functional import image_gradients

from ..geometry.geometryutils import transform_pointcloud
from ..geometry.se3utils import se3_exp, se3_hat
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .icputils import solve_linear_system

# The MaskedTensor module is still under development at the moment of this implementation, there will be changed in the future.
from torch.masked.maskedtensor import MaskedTensor

# For debug purpose
import open3d as o3d
import sophus
import numpy as np

__all__ = [
    "convert_rgb_to_gray",
    "construct_intrinsics_pyramid",
    "construct_pyramids",
    "build_pyramid",
    "solve_GaussNewton",
    "solve_linear_system",
    "direct_image_align",
]

def convert_rgb_to_gray(
    rgb: torch.Tensor,
) -> torch.tensor:
    r"""Convert rgb images to grayscale image which represents the image intensity.

    Args:
        rgb (torch.Tensor): Input rgb image (channel first)

    Returns:
        torch.tensor: Ouput grayscale image (channel first)
    
    Shape: 
        -rgb: :math:`(.., 3, H, W)`
        -Ouput: :math:`(..., 1, H, W)`
    """    
    if not torch.is_tensor(rgb):
        raise TypeError(
            "Expected rgb to be of type torch.Tensor. Got {0}.".format(type(rgb))
        )
    if not rgb.size(-3) == 3:
        raise ValueError(
            "Expected rgb has a channel first structure, i.e. has the shape [..., 3, H, W]. Got {0}.".format(rgb.size())
        )
    
    return rgb_to_grayscale(img=rgb, num_output_channels=1)

def construct_intrinsics_pyramid(
    intrinsics: torch.Tensor,
    num_pyr_levels: int = 3,
) -> List[torch.Tensor]:
    """Scale down the intrinsics iteratively and construct an intrinsics pyramid of specified number of levels

    Args:
        intrinsics (torch.Tensor): Original intrinsics
        numPyrLevels (int, optional): Specified number of levels. Defaults to 3.

    Returns:
        List[torch.Tensor]: Pyramid as a list containing intinsics for each level, the last one is of the most coarse level, the first one is the original.
    """
    if not torch.is_tensor(intrinsics):
        raise TypeError(
            "Expected intrinsics to be of type torch.Tensor. Got {0}.".format(type(intrinsics))
        )
    
    K_3x3 = intrinsics.clone().squeeze()[0:3, 0:3]
    intrinsics_pyramid = [K_3x3]
    for _ in range(num_pyr_levels - 1):
        new_intrinsics = intrinsics_pyramid[-1].clone()
        new_intrinsics[:2, :3] = new_intrinsics[:2, :3] * 0.5
        intrinsics_pyramid.append(new_intrinsics)   
    return intrinsics_pyramid


def downsmaple_image(image: torch.Tensor, ds_ratio: int=2, mode: str='avg'):
    """Downsample image

    Args:
        image (torch.Tensor): Original image
        ds_ratio (int, optional): Downsample ratio. Defaults to 2.
        mode (str): Available mode for downsampling are: 
            - 'avg": Average over neighborhood
            - 'bilinear': Bilinear interpolation

    Returns:
        torch.Tensor: Downsampled image
    
    Shape:
        - image: :math:`(N, L, C, H, W)`
        - Output: :math:`(N, L, C, H/ds_ratio, W/ds_ratio)`
    """    
    if mode == 'avg':
        image_ds = (image[..., 0::2, 0::2] + image[..., 0::2, 1::2] + image[..., 1::2, 0::2] + image[..., 1::2, 1::2]) / 4.
    elif mode == 'bilinear':
        pass
    
    return image_ds


def downsample_depth(depth: torch.Tensor, ds_ratio: int=2, mode: str='avg'):
    """Downsample image, ignoring the zero depth pixels

    Args:
        image (torch.Tensor): Original image
        ds_ratio (int, optional): Downsample ratio. Defaults to 2.
        mode (str): Available mode for downsampling are: 
            - 'avg": Average over neighborhood
            - 'bilinear': Bilinear interpolation

    Returns:
        torch.Tensor: Downsampled image
    
    Shape:
        - image: :math:`(N, L, C, H, W)`
        - Output: :math:`(N, L, C, H/ds_ratio, W/ds_ratio)`
    """    
    if mode == 'avg':
        depth_ds = torch.stack([depth[..., 0::2, 0::2], depth[..., 0::2, 1::2], depth[..., 1::2, 0::2], depth[..., 1::2, 1::2]], dim=-1)
        num_valid_depth = torch.count_nonzero(depth_ds, dim=-1)
        num_valid_depth[torch.where(num_valid_depth == 0)] = 1 # To avoid divid by 0

        depth_ds = torch.sum(depth_ds, dim=-1, dtype=depth.dtype) / num_valid_depth
    elif mode == 'bilinear':
        pass

    return depth_ds
    

def construct_RGBD_pyramids(
    rgbdimage: RGBDImages,
    num_pyr_levels: int = 3,
) -> Tuple[List[torch.Tensor]]:
    if not isinstance(rgbdimage, RGBDImages):
        raise TypeError(
            "Expected img to be of type gradslam.RGBDImages. Got {0}.".format(type(rgbdimage))
        )
    if rgbdimage.shape[1] != 1:
        raise ValueError(
            "Sequence length of rgbdimages must be 1, but was {0}.".format(
                rgbdimage.shape[1]
            )
        )

    if not rgbdimage.channels_first:
        rgbdimage.to_channels_first_()

    # torch convert rgb to gray require channel first structure
    intensity = convert_rgb_to_gray(rgbdimage.rgb_image)

    # Valid depths mask
    mask = rgbdimage.valid_depth_mask.squeeze(-1)
    # if rgbdimages.channels_first:
    #     mask = mask.expand(-1, -1, 3, -1, -1)
    rgb_pyramid = [rgbdimage.rgb_image.squeeze()]
    intensity_pyramid = [intensity.squeeze()]
    depth_pyramid = [rgbdimage.depth_image.squeeze()]
    
    for _ in range(num_pyr_levels - 1):
        rgb_pyramid.append(downsmaple_image(rgb_pyramid[-1]))
        intensity_pyramid.append(downsmaple_image(intensity_pyramid[-1]))
        depth_pyramid.append(downsample_depth(depth_pyramid[-1]))
        print(rgb_pyramid[-1].size()) # ([1, 1, 3, 240, 320]) ([1, 1, 3, 120, 160])

    print(rgb_pyramid[-1].size()) # ([1, 1, 3, 60, 80])

    return rgb_pyramid, intensity_pyramid, depth_pyramid


@dataclass
class RGBDPyramid:
    def __init__(
        self,
        rgbdimage: RGBDImages,
        num_pyr_levels: int=3,
    ):
        self.rgb_pyramid, self.intensity_pyramid, self.depth_pyramid = construct_RGBD_pyramids(rgbdimage=rgbdimage, num_pyr_levels=num_pyr_levels)
        self.K_pyramid = construct_intrinsics_pyramid(rgbdimage.intrinsics)


def depth_ambigious_backprojection(H: int, W: int, K: torch.Tensor):
    """Backproject the pixel coordinates to a 3D unit plane (depth is ambigious)

    Args:
        height (int): Height of image
        width (int): Width of image
        intrinsics (torch.Tensor): Intrinsics

    Returns:
        torch.Tensor: Points of unit plane in 3D space
    """
    device = K.device
    K = K.squeeze()       # ([4, 4])
    # Back-projection
    K_inv = torch.linalg.inv(K[:3, :3]) # shape([3, 3])
    # print('Inverse K: ', K_inv)
    us = torch.arange(W, device=device, dtype=K.dtype).view(1, W).expand(H, W)
    vs = torch.arange(H, device=device, dtype=K.dtype).view(H, 1).expand(H, W)
    ones = torch.ones((H, W), device=device, dtype=K.dtype)
    hom_pixels = torch.stack((us, vs, ones), dim=2)
    hom_pixels = hom_pixels.view(H, W, 3, 1) # shape ([H, W, 3, 1])

    points3D_unit = torch.matmul(K_inv, hom_pixels)  # ([H, W, 3, 1])

    return points3D_unit

def backprojection(points3D_unit: torch.Tensor, depth: torch.Tensor):
    *_, H, W = depth.shape
    points3D = depth.view(H, W, 1, 1) * points3D_unit
    return points3D # shape ([H, W, 3, 1])


def point_projection(points3D: torch.Tensor, H: int, W: int, K: torch.Tensor, T: torch.Tensor):
    device = points3D.device
    points3D = torch.cat((points3D, torch.ones((H, W, 1, 1), device=device)), dim=2) # Homogenous coordinate
    # print('point_projection: ', points3D.shape, points3D.dtype, T.dtype) # [H, W, 4, 1]
    points3D_warped = torch.matmul(T, points3D) # Homogenous coordinate
    points3D_warped = points3D_warped[:, :, :3, :]

    # 3D-2D projection
    pixel_warped = torch.matmul(K, points3D_warped)
    print(pixel_warped[:, :, :2, :].shape, pixel_warped[:, :, 2:3, :].shape)
    pixel_warped = pixel_warped[:, :, :2, :] / (pixel_warped[:, :, 2:3, :] + 1e-7)
    pixel_warped[:, :, 0, :] /= W -1
    pixel_warped[:, :, 1, :] /= H -1
    pixel_warped = (pixel_warped - 0.5) * 2

    return pixel_warped # [H, W, 2, 1]


def calc_residuals(
    pixel_warped: torch.Tensor,
    I_prev: torch.Tensor,
    I_curr: torch.tensor,
):
    *_, H, W = I_prev.shape
    print('snippet from previous intensity:')
    
    warped_i = F.grid_sample(I_prev.view(1, 1, H, W), pixel_warped.squeeze().unsqueeze(0), padding_mode='zeros', align_corners=True).squeeze()

    residuals = warped_i - I_curr
    return residuals


def calc_gradient(image: torch.Tensor):
    if not torch.is_tensor(image):
        raise TypeError(
            "image should be torch.Tensor. Got {}.".format(type(image))
    )

    *_, H, W = image.shape
    grad_x = (image[..., :, 2:] - image[..., :, :W-2]) * 0.5
    l_r_pad = nn.ZeroPad2d((1, 1, 0, 0))
    grad_x = l_r_pad(grad_x)
    
    grad_y = (image[..., 2:, :] - image[..., :H-2, :]) * 0.5
    t_b_pad = nn.ZeroPad2d((0, 0, 1, 1))
    grad_y = t_b_pad(grad_y)

    return grad_x, grad_y


def calc_Jacobian(
    I_prev: torch.Tensor, 
    K: torch.Tensor, 
    points3D_warped: torch.Tensor, 
    pixel_warped: torch.Tensor
):
    *_, H, W = I_prev.shape
    device = I_prev.device
    # Calculate gradients
    grad_x, grad_y = calc_gradient(I_prev) # [H, W], [H, W]
    grad_x_warped = F.grid_sample(grad_x.view(1, 1, H, W), pixel_warped.squeeze().unsqueeze(0), padding_mode='zeros', align_corners=True).squeeze() # [H, W]
    grad_y_warped = F.grid_sample(grad_y.view(1, 1, H, W), pixel_warped.squeeze().unsqueeze(0), padding_mode='zeros', align_corners=True).squeeze() # [H, W]
    
    J_I = torch.stack((grad_x_warped, grad_y_warped), dim=2).view(H, W, 1, 2) # [H, W, 1, 2]
    # print(J_I[:10, :10])

    # Construct Jacobian of the warping function J_W
    fx = K[0, 0]
    fy = K[1, 1]
    zeros = torch.zeros((H, W), device=device)
    points3D_warped = points3D_warped.squeeze()
    X = points3D_warped[:, :, 0]
    Y = points3D_warped[:, :, 1]
    Z = points3D_warped[:, :, 2]
    valid_Z = (Z > 0)
    print(valid_Z.shape, torch.sum(~valid_Z))
    J_W_11 = fx/Z
    J_W_13 = -fx * X/(Z * Z)
    J_W_14 = -fx * (X*Y)/(Z*Z)
    J_W_15 = fx * (1 + (X*X)/(Z*Z))
    J_W_16 = -fx * Y/Z

    J_W_22 = fy/Z
    J_W_23 = -fy * Y/(Z*Z)
    J_W_24 = -fy * (1 + (Y*Y)/(Z*Z))
    J_W_25 = fy * (X*Y)/(Z*Z)
    J_W_26 = fy * X/Z

    J_W_1 = torch.stack(
        [J_W_11, zeros, J_W_13, J_W_14, J_W_15, J_W_16], dim=2
    ).view(H, W, 1, 6)
    J_W_2 = torch.stack(
        [zeros, J_W_22, J_W_23, J_W_24, J_W_25, J_W_26], dim=2
    ).view(H, W, 1, 6)
    J_W = torch.cat([J_W_1, J_W_2], dim=2)
    # print('calc Jacobian nan values in J_W11: ', (torch.isnan(J_W_11)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_W13: ', (torch.isnan(J_W_13)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_W14: ', (torch.isnan(J_W_14)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_W15: ', (torch.isnan(J_W_15)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_W16: ', (torch.isnan(J_W_16)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_I: ', (torch.isnan(J_I)).nonzero(as_tuple=False))
    # print('calc Jacobian nan values in J_W: ', (torch.isnan(J_W)).nonzero(as_tuple=False))

    # J = J_I @ J_W
    # print(J_I, J_W)
    J = torch.matmul(J_I, J_W)

    return J


def solve_GaussNewton(
    I_prev: torch.Tensor, 
    d_prev: torch.Tensor, 
    I_curr: torch.Tensor, 
    d_curr: torch.Tensor, 
    K: torch.Tensor, 
    T: torch.Tensor, 
    num_iters: int=10
):
    # print("gauss newton input tensors shape:")
    # print('I_prev shape: ', I_prev.shape)
    # print('d_prev shape: ', d_prev.shape)
    # print('I_curr shape: ', I_curr.shape)
    # print('K: ', K, 'shape: ', K.shape)
    # print('T shape: ', T.shape)

    *_, H, W = I_curr.shape
    device = I_curr.device
    # Construct unit depth 3D point coordinates once since it's always the same for the same pyramid level
    points3D_unit = depth_ambigious_backprojection(H, W, K)

    err_prev = torch.inf
    T_prev = T

    for i in range(num_iters):
        # 2D-3D project the pixel coordinates in point coordinates in 3D space (camera coordinate system)
        cam_coord = backprojection(points3D_unit, d_curr)
        # 3D-2D projection
        pixel_warped = point_projection(cam_coord, H, W, K, T_prev)

        # Construct masks for valid entries
        valid_depth_mask = (cam_coord[:, :, 2] > 0).squeeze()

        pixel_warped = pixel_warped.clamp(-2, 2)
        valid_warped_pixel = pixel_warped.squeeze() < 1
        valid_warped_pixel = valid_warped_pixel * (pixel_warped.squeeze() > -1)
        valid_warped_pixel = valid_warped_pixel[:, :, 0] * valid_warped_pixel[:, :, 1]
        print(valid_warped_pixel.shape, torch.sum(valid_warped_pixel))
        
        # Calculate residuals
        r = calc_residuals(pixel_warped, I_prev, I_curr)

        # Calculate Jacobians
        J = calc_Jacobian(I_prev, K, cam_coord, pixel_warped)

        # Filter invalid entries
        valid_mask = valid_depth_mask * valid_warped_pixel
        r[~valid_mask] = 0.
        J[~valid_mask, :, :] = 0.
        # r = r.nan_to_num(nan=0.)
        # J = J.nan_to_num(nan=0.)

        # weighting
    
        # Solve Gauss-Newton
        J = J.view(-1, 6)
        r = r.view(-1, 1)
        Jt = J.T
        err = torch.sum(torch.matmul(r.T, r))

        b = torch.matmul(Jt, r)
        A = torch.matmul(Jt, J)
        inc = -torch.linalg.solve(A, b)
        print('gn solver, nan values in r: ', torch.sum(torch.isnan(r)))
        print('gn solver, nan values in J: ', torch.sum(torch.isnan(J)))
        print('{}. iteration, error = {}'.format(i+1, err))
        print(torch.sum(torch.isnan(b)), torch.sum(torch.isnan(A)))

        # Apply incremental to previous transformation
        T = torch.matmul(T_prev, se3_exp(inc))
        print(T)
        T_prev = T

        delta = torch.abs(err - err_prev)
        # if delta < 1e-6:
        #     break
        err_prev = err

    return T


def direct_image_align(
    curr_rgbd: RGBDImages,
    prev_rgbd: RGBDImages,
    initial_transform: torch.Tensor,
    num_pyr_levels: int = 3,
    num_iters: int = 20,
):
    if not curr_rgbd.channels_first:
        curr_rgbd.to_channels_first_()
    if not prev_rgbd.channels_first:
        prev_rgbd.to_channels_first_()

    device = curr_rgbd.device
    # print(curr_rgbd.depth_image.squeeze()[30:40, 30:40])

    curr_pyramids = RGBDPyramid(rgbdimage=curr_rgbd,
                                num_pyr_levels=num_pyr_levels)
    prev_pyramids = RGBDPyramid(rgbdimage=prev_rgbd, 
                                num_pyr_levels=num_pyr_levels)
    curr_I_pyr = curr_pyramids.intensity_pyramid
    curr_d_pyr = curr_pyramids.depth_pyramid
    prev_I_pyr = prev_pyramids.intensity_pyramid
    prev_d_pyr = prev_pyramids.depth_pyramid
    K_pyr = curr_pyramids.K_pyramid

    transform_prev = initial_transform # torch.eye(4, device=device)

    for i in range(1, num_pyr_levels + 1):
    # for i in range(1, 2):
        print("=========== {}. Pyramid ===========".format(num_pyr_levels+1 - i))
        I_curr = curr_I_pyr[-i].squeeze(0)
        d_curr = curr_d_pyr[-i].squeeze(0)
        I_prev = prev_I_pyr[-i].squeeze(0)
        d_prev = prev_d_pyr[-1].squeeze(0)
        K = K_pyr[-i].squeeze()
        
        transform = solve_GaussNewton(I_prev=I_prev, d_prev=d_prev, I_curr=I_curr, d_curr=d_curr, K=K, T=transform_prev, num_iters=num_iters)
        transform_prev = transform

        print('=====================')
        print("{}. pyramid level".format(i))
        print("Transformation = \n", transform)
        print('=====================')
    
    return transform

    

if __name__ == '__main__':
    # Test the implementation
    K = torch.eye(4).view(1, 1, 4, 4)
    K[0, 0] = K[1, 1] = 360
    K[0, 2] = 320
    K[1, 2] = 240
    K = K.view(1, 1, 4, 4)
    print(construct_intrinsics_pyramid(K))