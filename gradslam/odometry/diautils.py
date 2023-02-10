from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
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
    "intrinsics_pyrdown",
    "image_pyrdomn",
    "build_pyramid",
    "gauss_newton_solver",
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

def intrinsics_pyrdown(
    intrinsics: torch.Tensor,
    numPyrLevels: int = 3,
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
    intrinsics_pyramid = []
    intrinsics_pyramid.append(intrinsics)
    for _ in range(numPyrLevels - 1):
        new_intrinsics = intrinsics_pyramid[-1].clone()
        new_intrinsics[:, :, :2, :2] = new_intrinsics[:, :, :2, :2] * 0.5
        intrinsics_pyramid.append(new_intrinsics)
    
    return intrinsics_pyramid

def downsmaple_images():
    pass

# TODO check
def image_pyrdomn(
    rgbdimages: RGBDImages,
    numPyrLevels: int = 3,
) -> Tuple[List[torch.Tensor]]:
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected img to be of type gradslam.RGBDImages. Got {0}.".format(type(rgbdimages))
        )
    if rgbdimages.shape[1] != 1:
        raise ValueError(
            "Sequence length of rgbdimages must be 1, but was {0}.".format(
                rgbdimages.shape[1]
            )
        )

    if not rgbdimages.channels_first:
        rgbdimages.to_channels_first_()

    # torch convert rgb to gray require channel first structure
    gray_image = convert_rgb_to_gray(rgbdimages.rgb_image)
    B = len(rgbdimages) # Batch size of the rgbdimages
    ds_ratio = 2 # downsample ratio default as 2

    # Valid depths mask
    mask = rgbdimages.valid_depth_mask.squeeze(-1)
    # if rgbdimages.channels_first:
    #     mask = mask.expand(-1, -1, 3, -1, -1)
    rgb_pyramid = [rgbdimages.rgb_image]
    intensity_pyramid = [gray_image]
    depth_pyramid = [rgbdimages.depth_image]

    # for _ in range(numPyrLevels - 1):
    #     mask = mask[..., ::ds_ratio, ::ds_ratio]
    #     print(mask[0].expand(-1, 3, -1, -1).size())
    #     # Downsample points and normals
    #     # points = [
    #     #     rgbdimages.global_vertex_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
    #     #     for b in range(B)
    #     # ]
    #     # normals = [
    #     #     rgbdimages.global_normal_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
    #     #     for b in range(B)
    #     # ]
    #     print(rgb_pyramid[-1].size())
    #     print(rgb_pyramid[-1][0][..., :, ::ds_ratio, ::ds_ratio].size())
    #     print((intensity_pyramid[-1][0][..., ::ds_ratio, ::ds_ratio][mask[0]]).size())
    #     rgb_pyramid.append([
    #         rgb_pyramid[-1][b][..., ::ds_ratio, ::ds_ratio][mask[b].expand(-1, 3, -1, -1)]
    #         for b in range(B)
    #     ])

    #     intensity_pyramid.append(torch.stack([
    #         intensity_pyramid[-1][b][..., :, ::ds_ratio, ::ds_ratio][mask[b]]
    #         for b in range(B)
    #     ]))

    #     depth_pyramid.append(torch.stack([
    #         depth_pyramid[-1][b][..., :, ::ds_ratio, ::ds_ratio][mask[b]]
    #         for b in range(B)
    #     ]))
    
    # Use average pooling for downsampling
    avg_pool = nn.AvgPool2d(kernel_size=ds_ratio, stride=ds_ratio)
    for _ in range(numPyrLevels - 1):
        ds_rgb, ds_intensity, ds_depth = [], [], []
        for b in range(B):
            # Downsample rgb images 
            ds_rgb.append(avg_pool(rgb_pyramid[-1][b]))
            ds_intensity.append(avg_pool(intensity_pyramid[-1][b]))
            ds_depth.append(avg_pool(depth_pyramid[-1][b]))
        # print(rgb_pyramid[-1].size()) # ([1, 1, 3, 240, 320]) ([1, 1, 3, 120, 160])
        rgb_pyramid.append(torch.stack(ds_rgb, dim=0))
        intensity_pyramid.append(torch.stack(ds_intensity, dim=0))
        depth_pyramid.append(torch.stack(ds_depth, dim=0))
    # print(rgb_pyramid[-1].size()) # ([1, 1, 3, 60, 80])

    return rgb_pyramid, intensity_pyramid, depth_pyramid

@dataclass
class Pyramid:
    def __init__(
        self,
        rgb_pyramid: List[torch.Tensor],
        intensity_pyramid: List[torch.Tensor],
        depth_pyramid: List[torch.Tensor],
        intrinsics_pyramid: List[torch.Tensor],
    ):
        self.rgb_pyramid = rgb_pyramid
        self.intensity_pyramid = intensity_pyramid
        self.depth_pyramid = depth_pyramid
        self.intrinsics_pyramid = intrinsics_pyramid
        
    
def bilinear_interpolate(img, x, y, height, width):
    x0 = torch.floor(x).type(torch.LongTensor)
    x1 = x0 + 1
    
    y0 = torch.floor(y).type(torch.LongTensor)
    y1 = y0 + 1

    # x0 = torch.clamp(x0, 0, width - 1).type(torch.LongTensor)
    # x1 = torch.clamp(x1, 0, width - 1).type(torch.LongTensor)
    # y0 = torch.clamp(y0, 0, height - 1).type(torch.LongTensor)
    # y1 = torch.clamp(y1, 0, height - 1).type(torch.LongTensor)

    # Check if the warped points lie in the image
    nan = torch.tensor(torch.nan, device=img.device)
    if x0 < 0 or x0 >= width:
        return nan
    if x1 < 0 or x1 >= width:
        return nan
    if y0 < 0 or y0 >= height:
        return nan
    if y1 < 0 or y1 >= height:
        return nan
    
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return Ia*wa + Ib*wb + Ic*wc + Id*wd


def build_pyramid(
    rgbdimages: RGBDImages,
    numPyrLevels: int = 3,
) -> Pyramid:
    rgb_pyramid, intensity_pyramid, depth_pyramid = image_pyrdomn(rgbdimages, numPyrLevels)
    intrinsics_pyramid = intrinsics_pyrdown(rgbdimages.intrinsics, numPyrLevels)

    return Pyramid(rgb_pyramid, intensity_pyramid, depth_pyramid, intrinsics_pyramid)


def depth_ambigious_backprojection(
    height: int, 
    width: int, 
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """Backproject the pixel coordinates to a 3D unit plane (depth is ambigious)

    Args:
        height (int): Height of image
        width (int): Width of image
        intrinsics (torch.Tensor): Intrinsics

    Returns:
        torch.Tensor: Points of unit plane in 3D space
    """

    device = intrinsics.device
    intrinsics = intrinsics.squeeze()       # ([4, 4])

    # Back-projection
    K_inv = torch.linalg.inv(intrinsics[:3, :3]) # shape([3, 3])
    # print('Inverse K: ', K_inv)

    n_rows = torch.arange(width).view(1, width).repeat(height, 1).to(device)
    n_cols = torch.arange(height).view(height, 1).repeat(1, width).to(device)
    pixel = torch.stack((n_rows, n_cols, torch.ones(height, width, device=device)), dim=2) # shape ([H, W, 3])

    # Cache computed 3D points
    depth_ambigious_point3d = torch.matmul(pixel.view(height, width, 1, 3), K_inv.T).squeeze()  # ([H, W, 3])

    return depth_ambigious_point3d


def compute_residuals(
    curr_intensity: torch.Tensor,
    curr_depth: torch.Tensor,
    prev_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    transform: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute photometric error
    Compute the image alignment error (residuals). Takes in the previous 
    intensity image and first backprojects it to 3D to obtain a pointcloud. 
    This pointcloud is then rotated by an SE(3) transform "xi", and then
    projected down to the current image. After this step, an intensity
    interpolation step is performed and we compute the error between the 
    projected image and the actual current intensity image.

    While performing the residuals, also cache information to speedup Jacobian computation.

    Args:
        curr_intensity (torch.Tensor): current intensity 
        curr_depth (torch.Tensor): current depth
        prev_intensity (torch.Tensor): previous intensity
        intrinsics (torch.Tensor): intrinsic of the current pyramid
        transform (torch.Tensor): initial transformation used to compute the residual

    Returns:
        torch.Tensor: The residual of the warped intensity and the previous intensity
        torch.Tensor: The reconstruct 3D coordinates by reprojecting the pixel coordinates into the 3D space, cached for later compuation of Jacobian

    Shape:
        - curr_intensity: :math:`(1, 1, H, W)`
        - curr_depth: :math:`(1, 1, H, W)`
        - prev_intensity: :math:`(1, 1, H, W)`
        - intrinsics: :math:`(1, 1, 4, 4)`
        - transform: :math:`(4, 4)`
        - Output1: :math:`(H, W)`
        - Output2: :math:`(H, W, 3)`
    """
    # Input type check
    if not torch.is_tensor(curr_intensity):
        raise TypeError(
            "Expected pre_intensity to be of type torch.Tensor. Got {0}.".format(type(curr_intensity))
        )
    if not torch.is_tensor(curr_depth):
        raise TypeError(
            "Expected pre_depth to be of type torch.Tensor. Got {0}.".format(type(curr_depth))
        )
    if not torch.is_tensor(prev_intensity):
        raise TypeError(
            "Expected cur_intensity to be of type torch.Tensor. Got {0}.".format(type(prev_intensity))
        )
    if not torch.is_tensor(intrinsics):
        raise TypeError(
            "Expected intrinsics to be of type torch.Tensor. Got {0}.".format(type(intrinsics))
        )
    
    # print('compute residuals input tensor shape: ')
    # print('curr_int shape: ', curr_intensity.size())
    # print('curr_depth shape: ', curr_depth.size())
    # print('prev_int shape: ', prev_intensity.size())
    # print('K shape: ', intrinsics.size())
    # print('transform shape: ', transform.size())

    width = prev_intensity.size(-1)
    height = prev_intensity.size(-2)
    device = prev_intensity.device
    residuals = torch.zeros(height, width, device=device)


    curr_intensity = curr_intensity.squeeze() # ([H, W])
    curr_depth = curr_depth.squeeze()         # ([H, W])
    prev_intensity = prev_intensity.squeeze() # ([H, W])
    intrinsics = intrinsics.squeeze()       # ([4, 4])

    # Back-projection
    '''
    K_inv = torch.linalg.inv(intrinsics[:3, :3]) # shape([3, 3])

    n_rows = torch.arange(width).view(1, width).repeat(height, 1).to(device)
    n_cols = torch.arange(height).view(height, 1).repeat(1, width).to(device)
    pixel = torch.stack((n_rows, n_cols, torch.ones(height, width, device=device)), dim=0) # shape ([3, H, W])

    # Cache computed 3D points
    cache_point3d = torch.matmul(K_inv, pixel.view(3, -1))  # ([3, HxW])
    # TODO check
    # valid_depth_mask = (cur_depth != 0).view(-1)
    # print(cache_point3d.size())
    # cache_point3d = (cur_depth.view(1, -1)[:, valid_depth_mask] * cache_point3d[:, valid_depth_mask])
    # print(cache_point3d.size())
    cache_point3d = (curr_depth.view(1, -1) * cache_point3d) # ([3, HxW])
    cache_point3d = (torch.matmul(transform[:3, :3], cache_point3d) + transform[:3, 3].view(3, 1)) # shape ([3, HXW])

    warped_pixel = torch.matmul(intrinsics[:3, :3], cache_point3d) # shape ([3, HxW])

    # Change into tensor with shape of ([H, W, 3])
    cache_point3d = cache_point3d.T.view(height, width, 3)
    warped_pixel = warped_pixel.T.view(height, width, 3)

    warped_pixel = warped_pixel[:, :, :2] / warped_pixel[:, :, 2].unsqueeze(-1)
    '''

    c_point3d = torch.zeros(height, width, 3, device=device)

    # warped_intensity = torch.zeros_like(prev_intensity)
    residuals = torch.zeros(height, width, device=device)
    c = 0
    fx_inv = 1 / intrinsics[0, 0]
    fy_inv = 1 / intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    for v in range(height):
        for u in range(width):

            #######################################
            Z = curr_depth[v, u]
            X = fx_inv * Z * (u - cx)
            Y = fy_inv * Z * (v - cy)
            point3d_warped = torch.matmul(transform[:3, :3], torch.asarray([X, Y, Z], device=device)) + transform[:3, 3]
            c_point3d[v, u, :] = point3d_warped
            if point3d_warped[2] <= 0:
                c = c + 1
                continue

            pixel_warped = torch.matmul(intrinsics[:3, :3], point3d_warped)
            u_pi = pixel_warped[0] / pixel_warped[2]
            v_pi = pixel_warped[1] / pixel_warped[2]
            #######################################

            # TODO check if warped pixel coord exceed image range
            # i_warped = bilinear_interpolate(prev_intensity, warped_pixel[v, u, 0], warped_pixel[v, u, 1], height, width)
            i_warped = bilinear_interpolate(prev_intensity, u_pi, v_pi, height, width)

            if not torch.isnan(torch.Tensor([i_warped])):
                residuals[v, u] = i_warped - curr_intensity[v, u]

    # print("Non valid depth count: ", c)

    return residuals, c_point3d


def compute_Jacobian(
    prev_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    cached_points3d: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jacobian of the photometric error function utilizing the chain rule

    Args:
        prev_intensity (torch.Tensor): previous intensity
        intrinsics (torch.Tensor): intrinsics matrix
        cached_points3d (torch.Tensor): cached 3D coordinates by reprojecting the pixel coordinates into 3D space

    Returns:
        torch.Tensor: Jacobian of the photometric error function

    Shape: 
        - prev_intensity: math:`(1, 1, H, W)`
        - intrinsics: math:`(4, 4)`
        - cached_point3d: math:`(H, W, 3)`
        - Output: math:`(1, 6)`
    """

    # print('compute jacobian input tensor shape: ')
    # print('pre_int shape: ', prev_intensity.size())
    # print('K shape: ', intrinsics.size())
    # print('cached_point3d shape: ', cached_points3d.size())

    # TODO check
    # Compute image gradient
    grad_x, grad_y = image_gradients(prev_intensity)
    grad_x = grad_x.squeeze()
    grad_y = grad_y.squeeze()

    width = prev_intensity.size(-1)
    height = prev_intensity.size(-2)
    device = prev_intensity.device

    intrinsics = intrinsics.squeeze()
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Jacobian = torch.zeros(height, width, 6, device=device)
    c = 0
    for v in range(height):
        for u in range(width):
            X, Y, Z = cached_points3d[v, u]
            if Z <= 0:
                c += 1
                continue
            
            u_ = fx * X/Z + cx
            v_ = fy * Y/Z + cy
            dx = bilinear_interpolate(grad_x, u_, v_, height, width)

            if torch.isnan(dx):
                continue
            dy = bilinear_interpolate(grad_y, u_, v_, height, width)
            if torch.isnan(dy):
                continue
            J_img = torch.asarray([dx, dy], device=device).view(1, 2)
            J_w = torch.asarray(
                [[fx/Z, 0, -fx*X/(Z*Z), -fx*(X*Y)/(Z*Z), fx*(1 + (X*X)/(Z*Z)), -fx*Y/Z],
                [0, fy/Z, -fy*Y/(Z*Z), -fy*(1+(Y*Y)/(Z*Z)), fy*X*Y/(Z*Z), fy*X/Z]],
                device=device
            ).view(2, 6)
            Jacobian[v, u] = torch.matmul(J_img, J_w).view(1, 6)
    
    # print("Non valid depth count: ", c)

    return Jacobian

def gauss_newton_solver(
    cur_intensity: torch.Tensor,
    cur_depth: torch.Tensor,
    pre_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    transform: torch.Tensor,
    numiters: int = 20,
):
    # print("gauss newton input tensors shape:")
    # print('cur_int shape: ', cur_intensity.size())
    # print('cur_depth shape: ', cur_depth.size())
    # print('pre_int shape: ', pre_intensity.size())
    # print('K: ', intrinsics, 'shape: ', intrinsics.size())
    # print('transform shape: ', transform.size())

    err_prev = torch.inf
    transform_prev = transform
    for i in range(numiters):
        r, cached_points3d = compute_residuals(cur_intensity, cur_depth, pre_intensity, intrinsics, transform)

        J = compute_Jacobian(pre_intensity, intrinsics, cached_points3d)
        
        r = r.view(-1, 1)
        J = J.view(-1, 6)

        Jt = J.T

        # print('r shape: ', r.size())
        # print('r has nan values: ', torch.isnan(r).any())
        # print('J shape: ', J.size(), 'J.T shape: ', Jt.size())
        # print('J has nan values: ', torch.isnan(J).any())

        err = torch.sum(torch.matmul(r.T, r))

        b = torch.matmul(Jt, r)
        A = torch.matmul(Jt, J)

        # inc = - solve_linear_system(A, b)
        inc = - torch.linalg.solve(A, b)

        transform = torch.matmul(transform_prev, se3_exp(inc))
        # transform = torch.matmul(transform_prev, torch.from_numpy(sophus.SE3.exp(inc.cpu().numpy()).matrix().astype(np.float32)).cuda())

        transform_prev = transform
    
        # TODO
        # delta = np.abs(err - err_prev)
        # if delta < 1e-6:
        #     break
        err_prev = err
        print("{}. iteration, error = {}".format(i+1, err))
    
    return transform


def direct_image_align(
    cur_rgbd: RGBDImages,
    pre_rgbd: RGBDImages,
    initial_transform: torch.Tensor,
    numPyrLevels: int = 3,
    numiters: int = 20,
):
    if not cur_rgbd.channels_first:
        cur_rgbd.to_channels_first_()
    if not pre_rgbd.channels_first:
        pre_rgbd.to_channels_first_()

    device = cur_rgbd.device
    
    cur_pyramid = build_pyramid(cur_rgbd, numPyrLevels=numPyrLevels)
    pre_pyramid = build_pyramid(pre_rgbd, numPyrLevels=numPyrLevels)
    cur_intensity_pyr = cur_pyramid.intensity_pyramid
    pre_intensity_pyr = pre_pyramid.intensity_pyramid
    cur_depth_pyr = cur_pyramid.depth_pyramid
    intrinsics_pyr = cur_pyramid.intrinsics_pyramid

    transform_prev = initial_transform # torch.eye(4, device=device)

    for i in range(1, numPyrLevels + 1):
        print("=========== {}. Pyramid ===========".format(numPyrLevels+1 - i))
        cur_intensity = cur_intensity_pyr[-i].squeeze(0)
        cur_depth = cur_depth_pyr[-i].squeeze(0)
        pre_intensity = pre_intensity_pyr[-i].squeeze(0)
        intrinsics = intrinsics_pyr[-i].squeeze()
        transform = gauss_newton_solver(cur_intensity, cur_depth, pre_intensity, intrinsics, transform_prev, numiters=numiters)
        transform_prev = transform

        print("{}. Pyramid estimated transform: ".format(numPyrLevels+1 - i))
        print(transform)
    
    return transform

    

if __name__ == '__main__':
    # Test the implementation
    K = torch.eye(4).view(1, 1, 4, 4)
    K[0, 0] = K[1, 1] = 360
    K[0, 2] = 320
    K[1, 2] = 240
    K = K.view(1, 1, 4, 4)
    print(intrinsics_pyrdown(K))