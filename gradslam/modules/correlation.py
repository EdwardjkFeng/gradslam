import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrLayer:
    def __init__(self, fmap1, fmap2, num_levels=3):
        self.num_levels = num_levels
        self.corr_pyramid = []
        self.B, self.C, self.H, self.W = fmap1.shape
        
        self.device = fmap1.device
        self.corr = CorrLayer.global_cor(fmap1, fmap2)
        self.matches = self.lookup()
        self.opt_flow = self.correlation_cost()

        self.local_corr = CorrLayer.local_cor(fmap1, fmap2, 1)

    def forward():
        pass

    def backward():
        pass
    
    @staticmethod
    def global_cor(fr, fq):
        """Global correlation evaluates the pairwise similarities between all locations in the reference and query features maps.

        Args:
            fmap1 (torch.Tensor): reference feature map
            fmap2 (torch.Tensor): query feature map

        Returns:
            torch.Tensor: correlation volume capturing the similarities between all pairs of spatial locations.
        """
        B, C, H, W = fr.shape
        fr = fr.reshape(C, H*W)
        fq = fq.reshape(C, H*W)
        fr_norm = torch.linalg.vector_norm(fr, dim=0, keepdim=True)
        fq_norm = torch.linalg.vector_norm(fq, dim=0, keepdim=True)
        denorm = torch.matmul(fr_norm.T, fq_norm)
        corr = torch.matmul(fr.transpose(0, 1), fq) / (denorm + 1e-10)
        
        return corr.view(B, H, W, H, W)

    @staticmethod
    def local_cor(fr, fq, r: int=2):
        """Local correlation evaluates the pairwise similarities between all locations in the reference feature map and their neighborhoods in the query feature map.

        Args:
            fr (torch.Tensor): reference feature map
            fq (torch.Tensor): query feature map
            r (int): radius to query the correspondence 

        Returns:
            torch.Tensor: correlation volume capturing the similarities between all pairs of spatial locations.
        """
        B, C, H, W = fr.shape
        fr = fr.reshape(C, H, W)
        fq = fq.reshape(C, H, W)
        p2d = (r, r, r, r)
        fq_pad = F.pad(fq, p2d, "replicate")
        fr_norm = torch.linalg.vector_norm(fr, dim=0)
        print(fr_norm.shape)
        fq_norm = torch.linalg.vector_norm(fq_pad, dim=0)
        corr = torch.zeros(H, W, 2*r+1, 2*r+1)
        for i in range(H):
            for j in range(W):
                for k in range(2*r+1):
                    for l in range(2*r+1):
                        corr[i, j, k, l] = torch.mm(fr[:, i, j].view(C, -1).T, fq_pad[:, i+k, j+l].view(C, -1))
                        corr[i, j, k, l] = corr[i, j, k, l] / (fr_norm[i, j] * fq_norm[i+k, j+l] + 1e-10)
        print(corr.shape)

        return corr.unsqueeze(0)

    def local_lookup(self):
        """Lookup operator to retrieve the correspondence pixel coordinates in neighborhood."""
        pass


    def lookup(self):
        "Lookup operator to retrieve the correspondence pixel coordinates."
        matches = torch.zeros((self.B, self.H, self.W, 2)).to(self.device)
        corr_flatten = self.corr.view(self.B, self.H*self.W, -1)
        for b in range(self.B):
            corr_triu = torch.triu(corr_flatten[b], diagonal=1)
            _, match_idx = corr_triu.max(0)
            match_idx = match_idx.view(self.H, self.W)
            matches[b] = torch.stack([match_idx // self.W, match_idx % self.W], -1)
        print(matches.shape)
        return matches.int()

    def correlation_cost(self):
        y, x = torch.meshgrid(
            torch.arange(self.H).to(self.device),
            torch.arange(self.W).to(self.device),
            indexing="ij"
        )
        pix = torch.stack([x, y], dim=-1).view(1, self.H, self.W, 2)
        pix = pix.expand(self.B, -1, -1, -1)
        corr_cost = self.matches - pix
        return corr_cost
        

    # def correlation_np(fmap1, fmap2):
    #     f1 = fmap1.reshape(3, -1)
    #     f2 = fmap2.reshape(3, -1)
    #     f1_norm = np.linalg.norm(f1, axis=0)
    #     f2_norm = np.linalg.norm(f2, axis=0)
    #     denominator = np.matmul(f1_norm.reshape(-1, 1),  f2_norm.reshape(1, -1))
    #     print(f1_norm)
    #     print(denominator)
    #     corr = np.matmul(f1.T, f2) / (denominator + 1e-10)

    #     return corr


class CorrBase(nn.Module):
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
        offset_y, offset_x = torch.meshgrid([torch.arange(0, 2*self.max_disp + 1),
                                             torch.arange(0, 2*self.max_disp + 1)],
                                             indexing='ij')
        H, W = corr_volume.shape[-2:]
        B = corr_volume.shape[0]
        
        print(corr_volume.shape)
        corr_max, corr_index = torch.max(corr_volume, dim=1, keepdim=True)
        print(corr_index.view(H, W))
        v, u = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        output = torch.stack([v, u], dim=-1).to(self.device)
        output = output.view(1, H, W, 2).expand(B, -1, -1, -1)
        corr_index = corr_index.permute(0, 2, 3, 1)
        cell_size = 2 * self.max_disp + 1
        output[..., 0] += (corr_index // cell_size).squeeze(-1) - self.max_disp
        output[..., 1] += (corr_index % cell_size).squeeze(-1) - self.max_disp
        output[..., 0] = torch.where(output[..., 0] < 0, 0, output[..., 0])
        output[..., 0] = torch.where(output[..., 0] >= H, H-1, output[..., 0])
        output[..., 1] = torch.where(output[..., 1] < 0, 0, output[..., 1])
        output[..., 1] = torch.where(output[..., 1] >= W, W-1, output[..., 1])
        return output


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
    dataset = Cofusion(basedir=cofusion_path, sequences=sequences, seqlen=2, dilation=5, start=520, height=240, width=320, channels_first=True, return_object_mask=False)
    loader = DataLoader(dataset=dataset, batch_size=1)
    colors, depths, *_ = next(iter(loader))

    # fin = np.random.rand(1, 3, 2, 3)
    # fmap1 = torch.from_numpy(fin).to(device)
    # fmap2 = torch.from_numpy(fin).to(device)
    print(colors.shape, depths.shape)
    fmap = torch.cat([colors, depths], dim=2).squeeze(0).to(device)
    print(fmap.shape)
    colors = colors.squeeze(0).to(device)
    fmap1 = colors[0].unsqueeze(0)
    fmap2 = colors[1].unsqueeze(0)

    H, W = fmap1.shape[-2:]
    # corrlayer = CorrLayer(fmap1, fmap1)
    corrlayer = CorrBase(pad_size=2, kernel_size=1, max_displacement=2)
    # corr = corrlayer.corr
    # corr = corr.view(H*W, H*W)
    corr = corrlayer(fmap1, fmap2)
    corr_cost, index = torch.max(corr, dim=1, keepdim=True)
    corr_cost = corr_cost.squeeze()


    # lookup
    # tri = torch.triu(corr, diagonal=0)
    # y, x = torch.meshgrid(
    #     torch.arange(H).to(device),
    #     torch.arange(W).to(device),
    #     indexing="ij"
    # )
    # pix = torch.stack([x, y], dim=-1).view(-1, 2)
    # match_idx = torch.argmax(tri, dim=1)
    # pix[:, 0] = (match_idx / W).int()
    # pix[:, 1] = match_idx % W

    # pix = corrlayer.matches[0]
    pix = corrlayer.lookup(corr).squeeze()

    # # pix = pix.view(-1, 2)
    # warped = torch.zeros((H, W, 3))
    # # warped = torch.index_select(torch.index_select(colors[0], 1, pix[:, 0]), 2, pix[:, 1])
    # # print(warped.shape)
    # for i in range(H):
    #     for j in range(W):
    #         u = pix[i, j, 0] if torch.abs(pix[i, j, 0] - i) < 10 else 0
    #         v = pix[i, j, 1] if torch.abs(pix[i, j, 1] - j) < 10 else 0
    #         warped[i, j, :] = colors[1, :, int(u), int(v)]
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