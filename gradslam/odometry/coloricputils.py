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


def solve_linear_system_PSD(
    A: torch.Tensor, b: torch.Tensor, damp: Union[float, torch.Tensor] = 1e-8
):
    r"""Solves the normal equations of a linear system Ax = b, given the constraint matrix A and the coefficient vector
    b. Note that this solves the normal equations, not the linear system. That is, solves :math:`A^T A x = A^T b`,
    not :math:`Ax = b`.

    Args:
        A (torch.Tensor): The constraint matrix of the linear system.
        b (torch.Tensor): The coefficient vector of the linear system.
        damp (float or torch.Tensor): Damping coefficient to optionally condition the linear system (in practice,
            a damping coefficient of :math:`\rho` means that we are solving a modified linear system that adds a tiny
            :math:`\rho` to each diagonal element of the constraint matrix :math:`A`, so that the linear system
            becomes :math:`(A^TA + \rho I)x = b`, where :math:`I` is the identity matrix of shape
            :math:`(\text{num_of_variables}, \text{num_of_variables})`. Default: 1e-8

    Returns:
        torch.Tensor: Solution vector of the normal equations of the linear system

    Shape:
        - A: :math:`(\text{num_of_equations}, \text{num_of_variables})`
        - b: :math:`(\text{num_of_equations}, 1)`
        - Output: :math:`(\text{num_of_variables}, 1)`
    """
    if not torch.is_tensor(A):
        raise TypeError(
            "Expected A to be of type torch.Tensor. Got {0}.".format(type(A))
        )
    if not torch.is_tensor(b):
        raise TypeError(
            "Expected b to be of type torch.Tensor. Got {0}.".format(type(b))
        )
    if not (isinstance(damp, float) or torch.is_tensor(damp)):
        raise TypeError(
            "Expected damp to be of type float or torch.Tensor. Got {0}.".format(
                type(damp)
            )
        )
    if torch.is_tensor(damp) and damp.ndim != 0:
        raise ValueError(
            "Expected torch.Tensor damp to have ndim=0 (scalar). Got {0}.".format(
                damp.ndim
            )
        )
    if A.ndim != 2:
        raise ValueError("A should have ndim=2, but had ndim={}".format(A.ndim))
    if b.ndim != 2:
        raise ValueError("b should have ndim=2, but had ndim={}".format(b.ndim))
    if b.shape[1] != 1:
        raise ValueError("b.shape[1] should 1, but was {0}".format(b.shape[1]))
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            "A.shape[0] and b.shape[0] should be equal ({0} != {1})".format(
                A.shape[0], b.shape[0]
            )
        )
    # damp = (
    #     damp
    #     if torch.is_tensor(damp)
    #     else torch.tensor(damp, dtype=A.dtype, device=A.device)
    # )

    # # Construct the normal equations
    # A_t = torch.transpose(A, 0, 1)
    # damp_matrix = torch.eye(A.shape[0]).to(A.device)
    # A_At = torch.matmul(A, A_t) + damp_matrix * damp

    # # Solve the normal equations (for now, by inversion!)
    # return torch.matmul(A_t, torch.matmul(torch.inverse(A_At),  b))
    damp = (
        damp
        if torch.is_tensor(damp)
        else torch.tensor(damp, dtype=A.dtype, device=A.device)
    )

    # Construct the normal equations
    A_t = torch.transpose(A, 0, 1)
    damp_matrix = torch.eye(A.shape[1]).to(A.device)
    At_A = torch.matmul(A_t, A) + damp_matrix * damp

    # Avoid numerical instablitiy
    if torch.linalg.det(At_A) == 0:
        return torch.zeros(A.shape[1], 1, device=A.device)

    # Solve the normal equations (for now, by inversion!)
    return torch.matmul(torch.inverse(At_A), torch.matmul(A_t, b))



def computeColorGradient(tgt_pc, tgt_colors, tgt_normals):
    """Compute the gradient of the color of the continuous color funciton around the target point

    Args:
        tgt_pc (_type_): _description_
        tgt_colors (_type_): _description_
        tgt_normals (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    # tgt_pc = tgt_pc.contiguous()
    # tgt_colors = tgt_colors.contiguous()
    # tgt_normals = tgt_normals.contiguous()
    tgt_d_colors = torch.zeros(tgt_pc.size(), device=tgt_colors.device)

    _KNN = knn_points(tgt_pc, tgt_pc, K=4)
    dist, idx = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1)
    # DEBUG
    # print("index of nn size: ", idx.size())

    # distance threshold for knn
    dist_thresh = 0.05

    # dist_filter = (
    #     torch.ones(dist.size(1), dtype=torch.bool)
    #     if dist_thresh is None
    #     else torch.sum(dist[0], dim = 1) < dist_thresh
    # )
    # # DEBUG
    # # print(dist_filter.size(), tgt_pc.size())
    # print(torch.sum(dist[0], dim=1).size())
    # # print(dist_filter)

    # tgt_pc = tgt_pc[0, dist_filter, :]
    # tgt_colors = tgt_colors[0, dist_filter, :]
    # tgt_normals = tgt_normals[0, dist_filter, :]
    # DEBUG
    # print(tgt_pc.size())
    n_points = tgt_pc.shape[1]
    for i in range(n_points):
        dist_filter1 = (
            torch.ones(dist.size(2), dtype=torch.bool)
            if dist_thresh is None
            else dist[0, i] < dist_thresh
        )
        dist_filter2 = (
            torch.ones(dist.size(2), dtype=torch.bool)
            if dist_thresh is None
            else dist[0, i] > 1e-8
        )
        dist_filter = dist_filter1.logical_and(dist_filter2)
        valid_idx = idx[0, i, dist_filter]
        nn = valid_idx.size(0)
        # print(valid_idx, valid_idx.size())
        # DEBUG
        # print("number of nn: ", nn)
        if nn == 0:
            break
        A = torch.zeros(nn, 3, device=tgt_colors.device)
        b = torch.zeros(nn, 1, device=tgt_colors.device)
        vt = tgt_pc[0, i, :]
        intensity_t = torch.sum(tgt_colors[0, i, :]) / 3
        for j in range(nn):
            p_adj_idx = valid_idx[j]
            vt_adj = tgt_pc[0, p_adj_idx, :]
            intensity_t_adj = torch.sum(tgt_colors[0, p_adj_idx, :]) / 3
            A[j - 1, 0:3] = vt_adj - vt
            b[j - 1, 0] = intensity_t_adj - intensity_t
        
        A[nn - 1, 0:3] = (nn - 1) * tgt_normals[0, i, :]
        b[nn - 1, 0] = 0

        # # DEBUG
        # print(A.size(), A)

        sol = solve_linear_system_PSD(A, b).squeeze(-1)
        # DEBUG
        # print("color gradient: ", sol.is_cuda)

        tgt_d_colors[0, i, :] = sol

    return tgt_d_colors


def color_gauss_newton_solve(
    src_pc: torch.Tensor,
    src_colors: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_colors: torch.Tensor,
    tgt_normals: torch.Tensor,
    dist_thresh: Union[float, int, None] = None,
    lambda_geometric: Union[float, int]  = 0.968,
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
    if not (isinstance(lambda_geometric, float) 
            or isinstance(lambda_geometric, int)):
        raise TypeError(
            "Expected lambda_geometric to be of type float or int. Got {0}.".format(
                type(lambda_geometric)
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

    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
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
    A = torch.cat(
        [nx, ny, nz, nz * sy - ny * sz, nx * sz - nz * sx, ny * sx - nx * sy], 1
    )
    b = nx * (dx - sx) + ny * (dy - sy) + nz * (dz - sz)
    # DEBUG
    # print('A_geometric size: ', A_geometric.size(), 'b_geometric size: ', b_geometric.size())

    if lambda_geometric == 1.0:
        return A, b, chamfer_indices
    else:
        # DEBUG
        # print(lambda_geometric)
        src_colors = src_colors.contiguous()
        tgt_colors = tgt_colors.contiguous()

        # Constant weights to balance geometric and photometric terms
        lambda_photometric = 1 - lambda_geometric
        sqrt_lambda_geometric = math.sqrt(lambda_geometric)
        sqrt_lambda_photometric = math.sqrt(lambda_photometric)
        # DEBUG
        # print('lambda photometric: ', lambda_photometric)
        assoc_colors = torch.index_select(tgt_colors, 1, chamfer_indices)
            
        # Photometirc Jacobian and residuals
        i_s = torch.sum(src_colors, 2) / 3
        i_t = torch.sum(torch.index_select(tgt_colors, 1, chamfer_indices)[0, :, :].view(-1, 3), 1).unsqueeze(-1) / 3
        # DEBUG
        # print(i_s.size(), i_t.size())
        # d_i_t = computeColorGradient(tgt_pc, tgt_colors, tgt_normals)
        assoc_d_i_t = computeColorGradient(assoc_pts, assoc_colors, assoc_normals)[0, :, :].view(-1, 3)
        # DEBUG
        # print(tgt_colors.is_cuda, d_i_t.is_cuda)
        # assoc_d_i_t = torch.index_select(d_i_t, 1, chamfer_indices)[0, :, :].view(-1, 3)
        assoc_n = assoc_normals[0, :, :].view(-1, 3)

        vs = src_pc[0, dist_filter, :].view(-1, 3)
        vt = assoc_pts[0, :, :].view(-1, 3)
        # DEBUG
        # print(vs.size(), vt.size(), assoc_n.size())

        vs_proj = vs - torch.sum((vs - vt) * assoc_n, dim=1).unsqueeze(-1) * assoc_n
        is_proj = torch.sum(assoc_d_i_t * (vs_proj - vt), dim=1).unsqueeze(-1) + i_t[0, :].view(-1, 1)
        # DEBUG
        # print('vs_proj size: ', vs_proj.size(), 'is_proj size: ', is_proj.size())

        # M = torch.eye(assoc_pts.size(dim=1), device=src_pc.device) - torch.matmul(assoc_n, assoc_n.transpose(0, 1))
        # d_M = torch.matmul(assoc_d_i_t.transpose(0, 1), M)


        d_M = torch.zeros(assoc_d_i_t.size(), device=assoc_d_i_t.device)
        # print(d_M.size(), assoc_d_i_t.size(), assoc_n.size())
        for i in range(assoc_d_i_t.size(0)):
            d_M[i, :] = torch.matmul(assoc_d_i_t[i, :].view(1, -1), torch.eye(3, device=assoc_d_i_t.device) - torch.matmul(assoc_n[i, :].view(-1, 1), assoc_n[i, :].view(1, -1)))

        # DEBUG
        # print('d_M size: ', d_M.size())

        # dMx = d_M[0, :].view(-1, 1)
        # dMy = d_M[1, :].view(-1, 1)
        # dMz = d_M[2, :].view(-1, 1)
        dMx = d_M[:, 0].view(-1, 1)
        dMy = d_M[:, 1].view(-1, 1)
        dMz = d_M[:, 2].view(-1, 1)

        # DEBUG
        # print('dMx size: ', dMx.size(), 'sx size: ', sx.size())

        A_photometric = sqrt_lambda_photometric * torch.cat(
            [dMx, dMy, dMz, dMz * sy - dMy * sz, dMx * sz - dMz * sx, dMy * sx - dMx * sy], 1
        )
        b_photometric = - sqrt_lambda_photometric * (i_s[0, :].view(-1, 1) - is_proj)
        # DEBUG
        # print('A_photometric size: ', A_photometric.size(), 'b_photometric size: ', b_photometric.size())
        
        A = torch.cat([sqrt_lambda_geometric * A, A_photometric], 0)
        b = torch.cat([sqrt_lambda_geometric* b, b_photometric], 0)

    # DEBUG
    # print('A size: ', A.size(), 'b size: ', b.size())
    
    return A, b, chamfer_indices


def color_ICP(
    src_pc: torch.Tensor,
    src_colors: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_colors: torch.Tensor,
    tgt_normals: torch.Tensor,
    initial_transform: Optional[torch.Tensor] = None,
    numiters: int = 20,
    damp: float = 1e-8,
    dist_thresh: Union[float, int, None] = None,
    lambda_geometric: Union[float, int] = 0.968,
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
            src_pc, src_colors, tgt_pc, tgt_colors, tgt_normals, dist_thresh, lambda_geometric
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
            one_step_pc, src_colors, tgt_pc, tgt_colors, tgt_normals, dist_thresh, lambda_geometric
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


