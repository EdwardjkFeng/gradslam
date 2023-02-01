from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
from torchvision.transforms.functional import rgb_to_grayscale
from torchmetrics.functional import image_gradients

from ..geometry.geometryutils import transform_pointcloud
from ..geometry.se3utils import se3_exp, se3_hat
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .icputils import solve_linear_system

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
        rgb (torch.Tensor): Input rgb image

    Returns:
        torch.tensor: Ouput grayscale image
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
        rgbdimages = rgbdimages.to_channels_first(copy=True)

    gray_image = convert_rgb_to_gray(rgbdimages.rgb_image)
    B = len(rgbdimages)
    ds_ratio = 2
    # Valid depths mask
    mask = rgbdimages.valid_depth_mask.squeeze(-1)
    rgb_pyramid = [rgbdimages.rgb_image]
    intensity_pyramid = [gray_image]
    depth_pyramid = [rgbdimages.depth_image]

    for _ in range(numPyrLevels - 1):
        mask = mask[..., ::ds_ratio, ::ds_ratio]
        # Downsample points and normals
        # points = [
        #     rgbdimages.global_vertex_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
        #     for b in range(B)
        # ]
        # normals = [
        #     rgbdimages.global_normal_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
        #     for b in range(B)
        # ]
        rgb_pyramid.append([
            rgbdimages.rgb_image[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
            for b in range(B)
        ])

        intensity_pyramid.append([
            gray_image[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
            for b in range(B)
        ])

        depth_pyramid.append([
            rgbdimages.depth_image[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
            for b in range(B)
        ])

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
    x0 = torch.floor(x)
    x1 = x0 + 1
    
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    
    Ia = img[y0, x0][0]
    Ib = img[y1, x0][0]
    Ic = img[y0, x1][0]
    Id = img[y1, x1][0]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)


def build_pyramid(
    rgbdimages: RGBDImages,
    numPyrLevels: int = 3,
) -> Pyramid:
    rgb_pyramid, intensity_pyramid, depth_pyramid = image_pyrdomn(rgbdimages, numPyrLevels)
    intrinsics_pyramid = intrinsics_pyrdown(rgbdimages.intrinsics, numPyrLevels)

    return Pyramid(rgb_pyramid, intensity_pyramid, depth_pyramid, intrinsics_pyramid)


def compute_residuals(
    cur_intensity: torch.Tensor,
    cur_depth: torch.Tensor,
    pre_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    xi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute photometric error
    Compute the image alignment error (residuals). Takes in the previous intensity image and 
    first backprojects it to 3D to obtain a pointcloud. This pointcloud is then rotated by an 
    SE(3) transform "xi", and then projected down to the current image. After this step, an intensity
    interpolation step is performed and we compute the error between the projected image and the actual
    current intensity image.

    While performing the residuals, also cache information to speedup Jacobian computation.

    Shape:
        - cur_intensity: :math:`(1, 1, H, W)`
        - cur_depth: :math:`(1, 1, H, W)`
        - pre_intensity: :math:`(1, 1, H, W)`
        - intrinsics: :math:`(1, 1, 4, 4)`
    """
    # Input type check
    if not torch.is_tensor(cur_intensity):
        raise TypeError(
            "Expected pre_intensity to be of type torch.Tensor. Got {0}.".format(type(cur_intensity))
        )
    if not torch.is_tensor(cur_depth):
        raise TypeError(
            "Expected pre_depth to be of type torch.Tensor. Got {0}.".format(type(cur_depth))
        )
    if not torch.is_tensor(pre_intensity):
        raise TypeError(
            "Expected cur_intensity to be of type torch.Tensor. Got {0}.".format(type(pre_intensity))
        )
    if not torch.is_tensor(intrinsics):
        raise TypeError(
            "Expected intrinsics to be of type torch.Tensor. Got {0}.".format(type(intrinsics))
        )

    width = pre_intensity.size(-1)
    height = pre_intensity.size(-2)
    device = pre_intensity.device
    residuals = torch.zeros(height, width, device=device)


    cur_intensity = cur_intensity.squeeze()
    cur_depth = cur_depth.squeeze()
    pre_intensity = pre_intensity.squeeze()
    intrinsics = intrinsics.squeeze()

    # Back-projection
    K_inv = torch.linalg.inv(intrinsics[:3, :3])
    T = se3_exp(xi)

    n_rows = torch.arange(width).view(1, width).repeat(height, 1).to(device)
    n_cols = torch.arange(height).view(height, 1).repeat(1, width).to(device)
    pixel = torch.stack(n_rows, n_cols, torch.ones(height, width, device=device), dim=0)

    # Cache computed 3D points
    cache_point3d = torch.matmul(K_inv, pixel)
    # TODO check
    nonzero_depth = (cur_depth != 0)
    cache_point3d = (cur_depth[nonzero_depth] * cache_point3d).view(-1, 3).T
    cache_point3d = (torch.matmul(T[:3, :3], cache_point3d) + T[3, :3]).T.view(height, width, 3)

    warped_pixel = cache_point3d[:, :, :2] / cache_point3d[:, :, 3]

    warped_intensity = torch.zeros_like(pre_intensity)
    for v in range(height):
        for u in range(width):
            i_warped = bilinear_interpolate(cur_intensity, warped_pixel[v, u, 0], warped_pixel[v, u, 1], torch.t(width), torch.t(height))
            warped_intensity[v, u] = \
                i_warped if not torch.isnan(i_warped) else 0

    residuals = warped_intensity - pre_intensity

    return residuals, cache_point3d


def compute_Jacobian(
    pre_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    cached_points3d: torch.Tensor,
) -> torch.Tensor:

    # TODO check
    # Compute image gradient
    dx, dy = image_gradients(pre_intensity)
    dx = dx.squeeze()
    dy = dy.squeeze()

    width = pre_intensity.size(-1)
    height = pre_intensity.size(-2)
    device = pre_intensity.device

    intrinsics = intrinsics.squeeze()
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    Jacobian = torch.zeros(1, 6, device=device)
    for v in range(height):
        for u in range(width):
            X, Y, Z = cached_points3d[v, u, :]
            if Z <= 0: 
                break
            
            J_img = torch.asarray([dx[v, u], dy[v, u]], device=device).view(1, 2)
            J_w = torch.asarray(
                [[fx/Z, 0, -fx*X/(Z*Z), -fx*(X*Y)/(Z*Z), fx*(1 + (X*X)/(Z*Z)), -fx*Y/Z],
                [0, fy/Z, -fy*Y/(Z*Z), -fy*(1+(Y*Y)/(Z*Z)), fy*X*Y/(Z*Z), fy*X/Z]],
                device=device
            ).view(2, 6)
            Jacobian += torch.matmul(J_img, J_w).view(1, 6) \
                if not torch.isfinite(torch.matmul(J_img, J_w)[-1]) else 0

    return Jacobian

def gauss_newton_solver(
    cur_intensity: torch.Tensor,
    cur_depth: torch.Tensor,
    pre_intensity: torch.Tensor,
    intrinsics: torch.Tensor,
    xi: torch.Tensor,
    numiters: int = 20,
):
    xi_prev = xi

    err_prev = torch.inf
    for _ in range(numiters - 1):
        r, cached_points3d = compute_residuals(cur_intensity, cur_depth, pre_intensity, intrinsics, xi_prev)

        J = compute_Jacobian(pre_intensity, intrinsics, cached_points3d)

        Jt = J.T

        err = torch.sum(torch.matmul(r.T, r))

        b = torch.matmul(Jt, b)
        A = torch.matmul(Jt, J)

        inc = - solve_linear_system(A, b)
        xi_prev = xi
        xi = se3_hat(torch.matmul(se3_exp(xi), se3_exp(inc)))
    
        if (err / err_prev > 0.995):
            break
        err_prev = err
    
    return se3_exp(xi)


def direct_image_align(
    cur_rgbd: RGBDImages,
    pre_rgbd: RGBDImages,
    initial_transform: torch.Tensor,
    numPyrLevels: int = 3,
    numiters: int = 20,
):
    if not cur_rgbd.channels_first():
        cur_rgbd = cur_rgbd.to_channels_first()
    if not pre_rgbd.channels_first():
        pre_rgbd = pre_rgbd.to_channels_first()

    device = cur_rgbd.device
    
    cur_pyramid = build_pyramid(cur_rgbd, numPyrLevels=numPyrLevels)
    pre_pyramid = build_pyramid(pre_rgbd, numPyrLevels=numPyrLevels)
    cur_intensity_pyr = cur_pyramid[1]
    pre_intensity_pyr = pre_pyramid[1]
    cur_depth_pyr = cur_pyramid[2]
    intrinsics_pyr = cur_pyramid[3]

    transform = torch.eye(4, device=device)

    for i in range(1, numPyrLevels + 1):
        cur_intensity = cur_intensity_pyr[-i]
        cur_depth = cur_depth_pyr[-i]
        pre_intensity = pre_intensity_pyr[-i]
        intrinsics = intrinsics_pyr[-i]
        xi = se3_hat(transform)
        transform = gauss_newton_solver(cur_intensity, cur_depth, pre_intensity, intrinsics, xi, numiters=numiters)
    
    return transform

    

if __name__ == '__main__':
    # Test the implementation
    K = torch.eye(4).view(1, 1, 4, 4)
    K[0, 0] = K[1, 1] = 360
    K[0, 2] = 320
    K[1, 2] = 240
    K = K.view(1, 1, 4, 4)
    print(intrinsics_pyrdown(K))