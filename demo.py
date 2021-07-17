import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
import scipy.ndimage



def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

class PoissonEqn(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)

        # Compute gradient and laplacian
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)

        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
        self.laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)

        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels': self.pixels, 'grads': self.grads, 'laplace': self.laplace}



class NormalMap(Dataset):
    def __init__(self, normal_path):
        super().__init__()

        N = np.load(normal_path)
        Nx, Ny, Nz = N[:, :,  0], N[:, :,  1], N[:, :,  2]

        grads_x = - Nx / Nz
        grads_y = - Ny / Nz

        # Compute gradient and laplacian
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)

        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
        sidelength = N.shape
        self.coords = get_mgrid(151, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'grads': self.grads}


def gradients_mse(model_output, coords, gt_gradients):
    # compute gradients on the model
    gradients = gradient(model_output, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss


def image_demo():
    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True)
    img_siren.cuda()

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
            axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            coords = get_mgrid(2 ** 10, 1) * 5 * np.pi

            sin_1 = torch.sin(coords)
            sin_2 = torch.sin(coords * 2)
            sum = sin_1 + sin_2

            fig, ax = plt.subplots(figsize=(16, 2))
            ax.plot(coords, sum)
            ax.plot(coords, sin_1)
            ax.plot(coords, sin_2)
            plt.title("Rational multiple")
            plt.show()

            sin_1 = torch.sin(coords)
            sin_2 = torch.sin(coords * np.pi)
            sum = sin_1 + sin_2

            fig, ax = plt.subplots(figsize=(16, 2))
            ax.plot(coords, sum)
            ax.plot(coords, sin_1)
            ax.plot(coords, sin_2)
            plt.title("Pseudo-irrational multiple")
            plt.show()


def possion_demo():
    N_gt_path = r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\normal.npy'
    N_gt = np.load(N_gt_path)
    h, w, _ = N_gt.shape
    N_map_data = NormalMap(N_gt_path)
    dataloader = DataLoader(N_map_data, batch_size=1, pin_memory=True, num_workers=0)

    poisson_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                          hidden_layers=3, outermost_linear=True)
    poisson_siren.cuda()

    total_steps = 1000
    steps_til_summary = 50

    optim = torch.optim.Adam(lr=1e-4, params=poisson_siren.parameters())

    model_input, gt = next(iter(dataloader))
    gt = {key: value.cuda() for key, value in gt.items()}
    model_input = model_input.cuda()

    for step in range(total_steps):
        start_time = time.time()

        model_output, coords = poisson_siren(model_input)
        train_loss = gradients_mse(model_output, coords, gt['grads'])

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))

            img_grad = gradient(model_output, coords)
            zxzy = img_grad.cpu().view(h, w, 2).detach().numpy()
            zx = zxzy[:, :, 0]
            zy = zxzy[:, :, 1]
            N_est = np.array([zx, zy, np.ones_like(zx)]).transpose([1, 2, 0])
            N_est = N_est / np.linalg.norm(N_est, axis=1, keepdims=True)
            from hutils.PhotometricStereoUtil import evalsurfaceNormal
            Error_map, MAE, MedianE = evalsurfaceNormal(N_est, N_gt, np.ones_like(zx).astype(np.bool))
            print(MAE)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(h, w).detach().numpy())
            axes[1].imshow(N_est/2 + 0.5)
            axes[2].imshow(N_gt / 2 + 0.5)
            print(zxzy.shape)
            # axes[1].imshow()

            plt.show()

        optim.zero_grad()
        train_loss.backward()
        optim.step()



def possion_demo_ori():
    cameraman_poisson = PoissonEqn(128)
    dataloader = DataLoader(cameraman_poisson, batch_size=1, pin_memory=True, num_workers=0)

    poisson_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                          hidden_layers=3, outermost_linear=True)
    poisson_siren.cuda()

    total_steps = 1000
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=poisson_siren.parameters())

    model_input, gt = next(iter(dataloader))
    gt = {key: value.cuda() for key, value in gt.items()}
    model_input = model_input.cuda()

    for step in range(total_steps):
        start_time = time.time()

        model_output, coords = poisson_siren(model_input)
        train_loss = gradients_mse(model_output, coords, gt['grads'])

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))

            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(128, 128).detach().numpy())
            axes[1].imshow(img_grad.cpu().norm(dim=-1).view(128, 128).detach().numpy())
            axes[2].imshow(img_laplacian.cpu().view(128, 128).detach().numpy())
            plt.show()

        optim.zero_grad()
        train_loss.backward()
        optim.step()

if __name__ == "__main__":

    possion_demo()
