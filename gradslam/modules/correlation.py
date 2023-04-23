import torch
import torch.nn.functional as F

class CorrLayer:
    def __init__(self, fmap1, fmap2, num_levels=3):
        self.num_levels = num_levels
        self.corr_pyramid = []
        self.corr = CorrLayer.correlation(fmap1, fmap2)
        self.corr_np = CorrLayer.correlation_np(
            fmap1.cpu().detach().numpy(),
            fmap2.cpu().detach().numpy()
        )


    def forward():
        pass

    def backward():
        pass
    
    @staticmethod
    def correlation(fmap1, fmap2):
        B, C, H, W = fmap1.shape
        # fmap1 = torch.flatten(fmap1, start_dim=2)
        fmap1 = fmap1.reshape(C, H*W)
        fmap1_norm = torch.linalg.vector_norm(fmap1, dim=0, keepdim=True)
        print(fmap1_norm)

        fmap2 = fmap2.reshape(C, H*W)
        fmap2_norm = torch.linalg.vector_norm(fmap2, dim=0, keepdim=True)
        print(fmap1_norm.shape)
        denominator = torch.matmul(fmap1_norm.T, fmap2_norm)
        print(denominator)
        corr = torch.matmul(fmap1.transpose(0, 1), fmap2) / (denominator + 1e-10)
        
        return corr.view(B, H, W, H, W)

    def correlation_np(fmap1, fmap2):
        f1 = fmap1.reshape(3, -1)
        f2 = fmap2.reshape(3, -1)
        f1_norm = np.linalg.norm(f1, axis=0)
        f2_norm = np.linalg.norm(f2, axis=0)
        denominator = np.matmul(f1_norm.reshape(-1, 1),  f2_norm.reshape(1, -1))
        print(f1_norm)
        print(denominator)
        corr = np.matmul(f1.T, f2) / (denominator + 1e-10)

        return corr


if __name__ == "__main__":
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    import 

    device = torch.device("cuda")
    fin = np.random.rand(1, 3, 2, 3)
    fmap1 = torch.from_numpy(fin).to(device)
    fmap2 = torch.from_numpy(fin).to(device)

    H, W = fin.shape[-2:]
    corrlayer = CorrLayer(fmap1, fmap1)
    corr = corrlayer.corr
    corr = corr.view(H*W, H*W)

    out = corr.cpu().detach().numpy()
    fig = plt.figure()
    plt.imshow(out)
    plt.show()

    corr_np = corrlayer.corr_np
    plt.imshow(corr_np)
    plt.show()