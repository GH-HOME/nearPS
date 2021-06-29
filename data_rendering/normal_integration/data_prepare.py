import numpy as np
import torch

class SurfaceTent(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        x = np.linspace(-1, 1, num=sidelength)
        y = np.linspace(-1, 1, num=sidelength)
        XX, YY = np.meshgrid(x, y)
        YY = np.flip(YY, axis=0)
        slope = 0.6
        z = np.zeros_like(XX)
        zx = np.zeros_like(XX)
        zy = np.zeros_like(XX)
        mask_top = np.logical_and.reduce((XX>-0.8, XX<0.8, YY>0, YY<0.8))
        zy[mask_top] = -slope
        z[mask_top] = 0.8 * slope -slope * YY[mask_top]
        mask_bottom = np.logical_and.reduce((XX>-0.8, XX<0.8, YY<0, YY>-0.8))
        zy[mask_bottom] = slope
        z[mask_bottom] = 0.8 * slope + slope * YY[mask_bottom]
        # Compute gradient and laplacian
        grads_x, grads_y = torch.from_numpy(zx), torch.from_numpy(zy)
        xx, yy = torch.from_numpy(XX).float(), torch.from_numpy(YY.copy()).float()
        self.mask = np.ones_like(XX, bool)
        self.grads = torch.stack((grads_x, grads_y), dim=-1)[self.mask]
        self.coords = torch.stack((xx, yy), dim=-1)[self.mask]
        self.z = torch.from_numpy(z).double()
        n = normalize_normal_map(np.concatenate((-zx[..., None],
                                                 -zy[..., None],
                                                 np.ones_like(zx)[..., None]), axis=-1))
        self.n = camera_to_object(n)
        self.n_vis = (n + 1) / 2
        self.n_vis[~self.mask] = 1
        plt.imshow(self.n_vis)
        plt.show()
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return self.coords, {'depth': self.z, 'grads': self.grads}