import torch
import torch.nn.functional as F

class CorrLayer:
    def __init__(self, fmap1, fmap2, num_levels=3):
        self.num_levels = num_levels
        self.corr_pyramid = []
        self.corr = CorrLayer.correlation(fmap1, fmap2)


    def forward():
        pass

    def backward():
        pass
    
    @staticmethod
    def correlation(fmap1, fmap2):
        B, C, H, W = fmap1.shape
        fmap1 = fmap1.view(B, C, H*W)
        fmap1 = torch.flatten(fmap1, start_dim=2)
        fmap2 = fmap2.view(B, C, H*W)
        fmap1 = torch.flatten(fmap2, start_dim=2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)**2 / torch.matmul(fmap1.transpose(1, 2), fmap1) / torch.matmul(fmap2.transpose(1, 2), fmap2)
        print(corr.shape)
        
        return corr.view(B, H, W, H, W)


if __name__ == "__main__":
    from gradslam.datasets import Cofusion
    from gradslam import RGBDImages

    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2 as cv

    data_set = 'CoFusion'
    data_path = '/home/jingkun/Dataset/'
    cofusion_path = data_path + 'CoFusion/'

    sequences = ("room4-full",)
    # Load data
    dataset = Cofusion(basedir=cofusion_path, sequences=sequences, seqlen=2, dilation=10, start=540, height=60, width=80, channels_first=False, return_object_mask=False)
    loader = DataLoader(dataset=dataset, batch_size=1)
    colors, depths, *_ = next(iter(loader))
    device = torch.device("cuda")
    colors = colors.permute(0, 1, 4, 2, 3).to(device).squeeze(0)
    depths = depths.permute(0, 1, 4, 2, 3).to(device).squeeze(0)
    print(f"colors shape: {colors.shape}")  # torch.Size([2, 8, 240, 320, 3])
    print(f"depths shape: {depths.shape}")  # torch.Size([2, 8, 240, 320, 1])
    fmap = torch.cat([colors, depths], dim=1)
    print(f"feature map shape: {fmap.shape}")

    corr = CorrLayer(fmap[0:1, ...], fmap[1:2, ...]).corr.view(60*80, 60*80)
    corresp = torch.argmin(corr, dim=0).view(60, 80)
    y, x = torch.meshgrid(
        torch.arange(60).to(device).float(),
        torch.arange(80).to(device).float(),
        indexing="ij")
    pix = torch.stack([x, y], dim=-1)
    pix[:, :, 0] = (corresp / 80).int()
    pix[:, :, 1] = corresp % 80
    corr_pix = pix.cpu().detach().numpy().astype(int)
    print(corr_pix)
    
    img = colors[1].permute(1, 2, 0).cpu().detach().numpy()

    mimg = np.zeros_like(img)
    for i in range(60):
        for j in range(80):
            mimg[i, j, :] = img[corr_pix[i, j, 0], corr_pix[i, j, 1], :]
    
    mimg = colors[0].permute(1, 2, 0).cpu().detach().numpy()
    color = cv.normalize(mimg, None, norm_type=cv.NORM_MINMAX)
    fig = plt.figure()
    plt.imshow(color)
    plt.show()