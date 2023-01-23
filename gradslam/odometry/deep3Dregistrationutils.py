import torch

from deep_3d_registration.registration_network import Registration3dNetwork
from deep_3d_registration.inference import to_graph_pair

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

    del trained_model, network_output, graph_pair
    torch.cuda.empty_cache()

    return transform
