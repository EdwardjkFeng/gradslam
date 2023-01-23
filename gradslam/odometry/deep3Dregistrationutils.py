from typing import Optional, Union
import math

from chamferdist.chamfer import knn_points
import torch

from ..geometry.geometryutils import transform_pointcloud
from ..geometry.se3utils import se3_exp
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages

import numpy as np
import open3d as o3d
import torch
from ..deep_3d_registration.embedding_visualization import get_colored_point_cloud_feature
from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Data
from ..deep_3d_registration.registration_network import Registration3dNetwork
import pandas as pd
from ..deep_3d_registration.dataset import PairData
from ..deep_3d_registration.inference import to_graph_pair

__all__ = {
    "deep_3d_registration",
}


def load_trained_model(device):
    args = {
        "visualize_embeddings": False,
        "config": "indoor",
    }

    trained_model = Registration3dNetwork(config=args["config"], eval_mode=True, visualize_embeddings=args["visualize_embeddings"]).to(device)
    
    return trained_model


def deep_3d_registraiton(source_pcd, source_normals, target_pcd, target_normals, pose):
    source_pcd = source_pcd.contiguous()[0, :, :].view(-1, 3)
    source_normals = source_normals.contiguous()[0, :, :].view(-1, 3)
    target_pcd = target_pcd.contiguous()[0, :, :].view(-1, 3)
    target_normals = target_normals.contiguous()[0, :, :].view(-1, 3)

    source = torch.concatenate([source_pcd, source_normals], 1)
    target = torch.concatenate([target_pcd, target_normals], 1)

    device = source_pcd.device
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Create graph pair
    graph_pair = to_graph_pair(source, target, R, t)
    trained_model = load_trained_model(device)

    network_output = trained_model.forward(graph_pair)

    transform = torch.eye(4, device=device)
    transform[:3, :3] = network_output['R']
    transform[:3, 3] = network_output['t'].squeeze()

    return transform
