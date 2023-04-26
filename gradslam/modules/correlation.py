import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GlobalCorrLayer",
           "LocalCorrLayer"]

class GlobalCorrLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fr, fq):
        """Global correlation evaluates the pairwise similarities between all locations in the reference and query features maps.

        Args:
            fmap1 (torch.Tensor): reference feature map
            fmap2 (torch.Tensor): query feature map

        Returns:
            torch.Tensor: correlation volume capturing the similarities between all pairs of spatial locations.
        """
        B, C, H, W = fr.shape
        # fr_norm = torch.linalg.vector_norm(fr, dim=0, keepdim=True)
        # fq_norm = torch.linalg.vector_norm(fq, dim=0, keepdim=True)
        fr = F.normalize(fr).view(B, C, H*W)
        fq = F.normalize(fq).view(B, C, H*W)
        corr = torch.matmul(fr.transpose(1, 2), fq)
        print(corr.shape)
        return corr.view(B, H, W, H, W)
    
    def lookup(self, corr_volume):
        "Lookup operator to retrieve the correspondence pixel coordinates."
        B, H, W = corr_volume.shape[:3]
        matches = torch.zeros((B, H, W, 2)).to(corr_volume.device)
        corr_flatten = corr_volume.view(B, H*W, -1)
        corr_triu = torch.stack([torch.triu(corr_flatten[b], diagonal=1) for b in range(B)], dim=0)
        _, match_idx = corr_triu.max(dim=1)
        match_idx = match_idx.view(B, H, W)
        matches = torch.stack([match_idx // W, match_idx % W], -1)
        print(matches.shape)
        return matches.int()


class LocalCorrLayer(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1):
        if not kernel_size == 1:
            raise ValueError("kernel_size other than 1 is not implmented. Got {}".format(kernel_size))
        if not pad_size == max_displacement:
            raise ValueError("pad_size {} should be the same as max_displacement {}.".format(pad_size, max_displacement))
        if stride1 != 1 or stride2 != 1:
            raise ValueError("stide other than 1 is not implmented. Got stride1 = {} and stride2 {}".format(stride1, stride2))
        
        super().__init__()
        self.max_disp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)
        self.device = torch.device("cuda")

    def forward(self, fr, fq):
        """Apply the local correlaiton on two feature maps

        Args:
            fr (torch.Tensor): reference feature map
            fq (torch.Tensor): query feature map
        
        Return:
            torch.Tensor: correlaiton volume encoding that how similar a patch in reference map to its neighborhood in query fmap is.

        Shape:
            - fr: :math:`(B, C, H, W)`
            - fq: :math:`(B, C, H, W)`
            - output: :math:`(B, (2r^2 + 1), H, W)`
        """
        fq_pad = self.padlayer(fq)
        fr = F.normalize(fr)
        fq_pad = F.normalize(fq_pad)
        offset_y, offset_x = torch.meshgrid([torch.arange(0, 2*self.max_disp + 1),
                                             torch.arange(0, 2*self.max_disp + 1)],
                                             indexing='ij')
        
        H, W = fr.shape[-2:]
        output = torch.cat([
            torch.mean(fr * fq_pad[:, :, dy:H+dy, dx:W+dx], 1, keepdim=True)
            for dx, dy in zip(offset_x.reshape(-1), offset_y.reshape(-1))
            ], dim=1)
        return output

    def lookup(self, corr_volume):
        H, W = corr_volume.shape[-2:]
        B = corr_volume.shape[0]
        # offset_y, offset_x = torch.meshgrid([
        #     torch.arange(-self.max_disp,  self.max_disp+1),
        #     torch.arange(-self.max_disp,  self.max_disp+1)
        #     ], indexing='ij')

        _, corr_index = torch.max(corr_volume, dim=1, keepdim=True)
        v, u = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        output = torch.stack([v, u], dim=-1).to(self.device)
        output = output.view(1, H, W, 2).expand(B, -1, -1, -1)
        corr_index = corr_index.permute(0, 2, 3, 1)
        cell_size = 2 * self.max_disp + 1
        output[..., 0] += (corr_index // cell_size).squeeze(-1) - self.max_disp
        output[..., 1] += (corr_index % cell_size).squeeze(-1) - self.max_disp  
        output = self.bound_index(output, H, W)
        return output
    
    def bound_index(self, index, H, W):
        index[..., 0] = torch.where(index[..., 0] < 0, 0, index[..., 0])
        index[..., 0] = torch.where(index[..., 0] >= H, H-1, index[..., 0])
        index[..., 1] = torch.where(index[..., 1] < 0, 0, index[..., 1])
        index[..., 1] = torch.where(index[..., 1] >= W, W-1, index[..., 1])
        return index


if __name__ == "__main__":
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    from gradslam.datasets import Cofusion
    from torch.utils.data import DataLoader

    device = torch.device("cuda")

    data_set = 'CoFusion'
    data_path = '/home/jingkun/Dataset/'
    cofusion_path = data_path + data_set + '/'
    sequences = ("room4-full",)
    dataset = Cofusion(basedir=cofusion_path, sequences=sequences, seqlen=2, dilation=5, start=600, height=30, width=40, channels_first=True, return_object_mask=False)
    loader = DataLoader(dataset=dataset, batch_size=1)
    colors, depths, *_ = next(iter(loader))

    # fin = np.random.rand(1, 3, 2, 3)
    # fmap1 = torch.from_numpy(fin).to(device)
    # fmap2 = torch.from_numpy(fin).to(device)
    fmap = torch.cat([colors, depths], dim=2).squeeze(0).to(device)
    colors = colors.squeeze(0).to(device)
    fmap1 = fmap[0].unsqueeze(0)
    fmap2 = fmap[1].unsqueeze(0)

    H, W = fmap1.shape[-2:]
    # corrlayer = GlobalCorrLayer()
    corrlayer = LocalCorrLayer(pad_size=1, kernel_size=1, max_displacement=1)
    corr = corrlayer(fmap1, fmap2)
    corr_cost, index = torch.max(corr, dim=1, keepdim=True)
    corr_cost = corr_cost.squeeze()

    pix = corrlayer.lookup(corr).squeeze()
    warped = torch.zeros((H, W, 3)).to(device)
    for i in range(H):
        for j in range(W):
            v = pix[i, j, 0]
            u = pix[i, j, 1]
            warped[i, j, :] = colors[0, :, int(v), int(u)]

    fig = plt.figure()
    img1 = cv.normalize(colors[0].permute(1, 2, 0).cpu().detach().numpy(), None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img1)
    plt.show()
    img2 = cv.normalize(colors[1].permute(1, 2, 0).cpu().detach().numpy(), None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img2.astype(float))
    plt.show()
    img = cv.normalize(corr_cost.cpu().detach().numpy(), None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img)
    plt.show()
    img = cv.normalize(warped.cpu().detach().numpy(), None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img)
    plt.show()