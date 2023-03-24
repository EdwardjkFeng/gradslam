"""
Bounding box estimation utility functions
"""

from typing import Optional

import torch

from ..structures.pointclouds import Pointclouds

def get_bouding_box_via_PCA(pcd: Pointclouds):
    points = pcd.points_list[0]
    center = torch.mean(points, 0, True)
    print(center.shape)

    points_mf = points - center

    cov = torch.matmul(points_mf.T, points_mf) / points_mf.shape[0]
    
    # Eigenvalue decomposition
    L, V = torch.linalg.eig(cov)
    V = V.real

    # Axes of the bounding box
    # v1 = V[0:3, 0]
    # v2 = V[0:3, 1]
    # v3 = V[0:3, 2]

    # Boundaries on each axis
    points_rotated = torch.matmul(points_mf, V)
    maxima = torch.argmax(points_rotated, dim=0)
    minima = torch.argmin(points_rotated, dim=0)
    print(minima, maxima)
    x_max = points[maxima, 0, 0]
    y_max = points[maxima, 1, 1]
    z_max = points[maxima, 2, 2]
    x_min = points[minima, 0, 0]
    y_min = points[minima, 1, 1]
    z_min = points[minima, 2, 2]

    print(x_max)

    corners = torch.stack([torch.stack([x_min, y_min, z_min]),
               torch.stack([x_min, y_min, z_max]),
               torch.stack([x_min, y_max, z_min]),
               torch.stack([x_min, y_max, z_max]),
               torch.stack([x_max, y_min, z_min]),
               torch.stack([x_max, y_min, z_max]),
               torch.stack([x_max, y_max, z_min]),
               torch.stack([x_max, y_max, z_max])], dim=1)
    
    print(corners.shape)
    
    return corners


