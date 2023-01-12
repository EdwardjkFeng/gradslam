from typing import Optional, Union
import math

from chamferdist.chamfer import knn_points
import torch

from ..geometry.geometryutils import transform_pointcloud
from ..geometry.se3utils import se3_exp
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages

from .icputils import solve_linear_system

__all__ = {
    "color_gauss_newton_solve",
    "color_ICP"
}


def computeColorGradient(tgt_pc, tgt_colors, tgt_normals):
    """Compute the gradient of the color of the continuous color funciton around the target point

    Args:
        tgt_pc (_type_): _description_
        tgt_colors (_type_): _description_
        tgt_normals (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    tgt_colors = tgt_colors.contiguous()
    tgt_d_colors = torch.zeros(tgt_pc.size())

    _KNN = knn_points(tgt_colors, tgt_colors)
    dist, idx = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1)

    # distance threshold for knn
    dist_thresh = 1

    dist_filter = (
        torch.ones_like(dist[0], dtype=torch.bool)
        if dist_thresh is None
        else dist[0] < dist_thresh
    )

    n_points = tgt_pc.shape[1]
    for i in range(n_points):
        nn = idx.size(dim=2)
        if nn == 0:
            break
        A = torch.zeros(nn, 3)
        b = torch.zeros(nn, 1)
        vt = tgt_pc[0, i, :]
        intensity_t = torch.sum(tgt_colors[0, i, :]) / 3
        for j in range(nn):
            p_adj_idx = idx[0, i, j]
            vt_adj = tgt_pc[0, j, :]
            intensity_t_adj = torch.sum(tgt_colors[0, j, :]) / 3
            A[j - 1, 0:3] = vt_adj - vt
            b[j - 1, 0] = intensity_t_adj - intensity_t
        
        A[nn - 1, :] = (nn - 1) * tgt_normals[0, i, :]
        b[nn - 1, :] = 0

        tgt_d_colors[0, i, :] = solve_linear_system(A, b)

    return tgt_d_colors


def color_gauss_newton_solve(
    src_pc: torch.Tensor,
    src_colors: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_colors: torch.Tensor,
    tgt_normals: torch.Tensor,
    dist_thresh: Union[float, int, None] = None,
):
    r"""Computes Gauss Newton step by forming linear equation. Points from `src_pc` which have a distance greater
    than `dist_thresh` to the closest point in `tgt_pc` will be filtered.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        src_colors (torch.Tensor): Per-point color vectors for each point in the source pointcloud.
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_colors (torch.Tensor): Per-point color vectors for each point in the target pointcloud.
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None

    Returns:
        tuple: tuple containing:

        - A (torch.Tensor): linear system equation
        - b (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc`
            that was not filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - src_colors: :math:`(1, N_t, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_colors: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - A: :math:`(N_sf, 6)` where :math:`N_sf \leq N_s`
        - b: :math:`(N_sf, 1)` where :math:`N_sf \leq N_s`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \leq N_s`
    """
    if not torch.is_tensor(src_pc):
        raise TypeError(
            "Expected src_pc to be of type torch.Tensor. Got {0}.".format(type(src_pc))
        )
    if not torch.is_tensor(src_colors):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(type(src_colors))
        )
    if not torch.is_tensor(tgt_pc):
        raise TypeError(
            "Expected tgt_pc to be of type torch.Tensor. Got {0}.".format(type(tgt_pc))
        )
    if not torch.is_tensor(tgt_colors):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(type(tgt_colors))
        )
    if not torch.is_tensor(tgt_normals):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(
                type(tgt_normals)
            )
        )
    if not (
        isinstance(dist_thresh, float)
        or isinstance(dist_thresh, int)
        or dist_thresh is None
    ):
        raise TypeError(
            "Expected dist_thresh to be of type float or int. Got {0}.".format(
                type(dist_thresh)
            )
        )
    if src_pc.ndim != 3:
        raise ValueError(
            "src_pc should have ndim=3, but had ndim={}".format(src_pc.ndim)
        )
    if tgt_pc.ndim != 3:
        raise ValueError(
            "tgt_pc should have ndim=3, but had ndim={}".format(tgt_pc.ndim)
        )
    if tgt_normals.ndim != 3:
        raise ValueError(
            "tgt_normals should have ndim=3, but had ndim={}".format(tgt_normals.ndim)
        )
    if src_pc.shape[0] != 1:
        raise ValueError(
            "src_pc.shape[0] should be 1, but was {} instead".format(src_pc.shape[0])
        )
    if tgt_pc.shape[0] != 1:
        raise ValueError(
            "tgt_pc.shape[0] should be 1, but was {} instead".format(tgt_pc.shape[0])
        )
    if tgt_normals.shape[0] != 1:
        raise ValueError(
            "tgt_normals.shape[0] should be 1, but was {} instead".format(
                tgt_normals.shape[0]
            )
        )
    if tgt_pc.shape[1] != tgt_normals.shape[1]:
        raise ValueError(
            "tgt_pc.shape[1] and tgt_normals.shape[1] must be equal. Got {0}!={1}".format(
                tgt_pc.shape[1], tgt_normals.shape[1]
            )
        )
    if src_pc.shape[2] != 3:
        raise ValueError(
            "src_pc.shape[2] should be 3, but was {} instead".format(src_pc.shape[2])
        )
    if tgt_pc.shape[2] != 3:
        raise ValueError(
            "tgt_pc.shape[2] should be 3, but was {} instead".format(tgt_pc.shape[2])
        )
    if tgt_normals.shape[2] != 3:
        raise ValueError(
            "tgt_normals.shape[2] should be 3, but was {} instead".format(
                tgt_normals.shape[2]
            )
        )

    # Constant weights to balance geometric and photometric terms
    lambda_geometric = 0.968
    lambda_photometric = 1 - lambda_geometric
    sqrt_lambda_geometric = math.sqrt(lambda_geometric)
    sqrt_lambda_photometric = math.sqrt(lambda_photometric)

    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    src_colors = src_colors.contiguous()
    tgt_colors = tgt_colors.contiguous()
    tgt_normals = tgt_normals.contiguous()

    _KNN = knn_points(src_pc, tgt_pc)
    dist1, idx1 = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1)

    dist_filter = (
        torch.ones_like(dist1[0], dtype=torch.bool)
        if dist_thresh is None
        else dist1[0] < dist_thresh
    )
    chamfer_indices = idx1[0][dist_filter].long()

    sx = src_pc[0, dist_filter, 0].view(-1, 1)
    sy = src_pc[0, dist_filter, 1].view(-1, 1)
    sz = src_pc[0, dist_filter, 2].view(-1, 1)

    # Closest point/normal to each source point
    assoc_pts = torch.index_select(tgt_pc, 1, chamfer_indices)
    assoc_normals = torch.index_select(tgt_normals, 1, chamfer_indices)

    # Closest destination point to each source point
    dx = assoc_pts[0, :, 0].view(-1, 1)
    dy = assoc_pts[0, :, 1].view(-1, 1)
    dz = assoc_pts[0, :, 2].view(-1, 1)

    nx = assoc_normals[0, :, 0].view(-1, 1)
    ny = assoc_normals[0, :, 1].view(-1, 1)
    nz = assoc_normals[0, :, 2].view(-1, 1)
    
    # Geometric Jacobian and residuals
    A_geometric = sqrt_lambda_geometric * torch.cat(
        [nx, ny, nz, nz * sy - ny * sz, nx * sz - nz * sx, ny * sx - nx * sy], 1
    )
    b_geometric = sqrt_lambda_geometric * (nx * (dx - sx) + ny * (dy - sy) + nz * (dz - sz))

    # Photometirc Jacobian and residuals
    i_s = torch.sum(src_colors, 2) / 3
    i_t = torch.sum(tgt_colors, 2) / 3
    d_i_t = computeColorGradient(tgt_pc, tgt_colors, tgt_normals)
    assoc_d_i_t = torch.index_select(d_i_t, 1, chamfer_indices)[0, :, :].view(-1, 3)
    assoc_n = assoc_normals[0, :, :].view(-1, 3)

    vs = src_pc[0, dist_filter, :].view(-1, 3)
    vt = assoc_pts[0, :, :].view(-1, 3)
    vs_proj = vs - torch.matmul(torch.dot((vs - vt), assoc_n), assoc_n)
    is_proj = assoc_d_i_t.dot(vs_proj - vt) + i_t[0, :, 0].view(-1, 1)

    M = torch.eye(assoc_pts.size(dim=1)) - torch.matmul(assoc_n, assoc_n.transpose(0, 1))
    d_M = torch.matmul(assoc_d_i_t.transpose(0, 1), M)

    dMx = dMx[:, 0].view(-1, 1)
    dMy = dMx[:, 0].view(-1, 1)
    dMz = dMx[:, 0].view(-1, 1)

    A_photometric = sqrt_lambda_photometric * torch.cat(
        [dMx, dMy, dMz, dMz * sy - dMy * sz, dMx * sz - dMz * sx, dMy * sx - dMx * sy], 1
    )
    b_photometric = sqrt_lambda_photometric * (is_proj - i_s[0, :, 0].view(-1, 1))
    
    A = torch.cat([A_geometric, A_photometric], 1)
    b = torch.cat([b_geometric, b_photometric], 1)
    
    return A, b, chamfer_indices


def color_ICP(
    src_pc: torch.Tensor,
    src_colors: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    tgt_colors: torch.Tensor,
    initial_transform: Optional[torch.Tensor] = None,
    numiters: int = 20,
    damp: float = 1e-8,
    dist_thresh: Union[float, int, None] = None,
):
    """Computes a rigid transformation between 'tgt_pc' (target pointcloud) and 'src_pc' (source pointcloud) using a 
    point-to-point error metric and the LM (Levenberg-Marquardt) solver.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        src_colors (torch.Tensor): Per-point color vectors for each point in the source pointcloud.
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        tgt_colors (torch.Tensor): Per-point color vectors for each point in the target pointcloud.
        initial_transform (torch.Tensor or None): The initial estimate of the transformation between 'src_pc'
            and 'tgt_pc'. If None, will use the identity matrix as the initial transform. Default: None
        numiters (int, optional): Number of iterations to run the optimization for. Default: 20
        damp (float, optional): Damping coefficient for nonlinear least-squares. Default: 1e-8
        dist_thresh (Union[float, int, None], optional): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None
    
    Returns:
        tuple: tuple containing:

        - transform (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in 'tgt_pc' for each point in 'src_pc' that was not 
          filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - src_colors: :math:`(1, N_t, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_colors: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - initial_transform: :math:`(4, 4)`
        - transform: :math:`(4, 4)`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \leq N_s`
    """
    if not torch.is_tensor(src_pc):
        raise TypeError(
            "Expected src_pc to be of type torch.Tensor. Got {0}.".format(type(src_pc))
        )
    if not torch.is_tensor(src_colors):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(type(src_colors))
        )
    if not torch.is_tensor(tgt_pc):
        raise TypeError(
            "Expected tgt_pc to be of type torch.Tensor. Got {0}.".format(type(tgt_pc))
        )
    if not torch.is_tensor(tgt_normals):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(type(tgt_normals))
        )
    if not torch.is_tensor(tgt_colors):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(type(tgt_colors))
        )
    if not (torch.is_tensor(initial_transform) or initial_transform is None):
        raise TypeError(
            "Expected initial_transform to be of type torch.Tensor. Got {0}.".format(type(initial_transform))
        )
    if not isinstance(numiters, int):
        raise TypeError(
            "Expected numiters to be of type int. Got{0}.".format(type(numiters))
        )
    if initial_transform.ndim != 2:
        raise ValueError(
            "Expected initial_transform.ndim to be 2. Got {0}.".format(initial_transform.ndim)
        )
    if not (initial_transform.shape[0] == 4 and initial_transform.shape[1] ==4):
        raise ValueError(
            "Expected initial_transform.shape to be (4, 4). Got {0}.".format(initial_transform.shape)
        )
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    src_colors = src_colors.contiguous()
    tgt_colors = tgt_colors.contiguous()
    tgt_normals = tgt_normals.contiguous()
    dtype = src_pc.dtype
    device = src_pc.device
    damp = torch.tensor(damp, dtype=dtype, device=device)

    # Include initial transform
    initial_transform = (
        torch.eye(4, dtype=dtype, device=device)
        if initial_transform is None
        else initial_transform
    )
    # transform_pointcloud from icputils.py
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform

    for it in range(numiters):
        # From the linear system and compute the residual
        A, b, chamfer_indices = color_gauss_newton_solve(
            src_pc, src_colors, tgt_pc, tgt_colors, tgt_normals, dist_thresh
        )
        residual = b[:, 0]

        # Solve the linear system
        xi = solve_linear_system(A, b, damp)

        # Apply exponential to find the transform
        residual_transform = se3_exp(xi)

        # Find error
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))

        # Lookahead error (for LM)
        # Calculate transformed cloud
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)

        # Form new linear system and compute one-step residual
        _, one_step_b, chamfer_indices_onestep = color_gauss_newton_solve(
            one_step_pc, src_colors, tgt_pc, tgt_colors, tgt_normals, dist_thresh
        )
        one_step_residual = one_step_b[:, 0]

        # Find new error
        new_err = torch.dot(one_step_residual.t(), one_step_residual)

        if new_err < err:
            # We are in a trust region
            src_pc = one_step_pc 
            damp = damp / 2

            # Update transform 
            transform = torch.mm(residual_transform, transform)

        else:
            damp = damp * 2

    return transform, chamfer_indices


