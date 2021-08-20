# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from datetime import datetime
import dataio, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial
import torch
from torchvision.transforms import ToTensor
import numpy as np

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='test_inpaint', required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=20000,
               help='Number of epochs to train for.')
p.add_argument('--k1', type=float, default=1, help='weight on prior')
p.add_argument('--sparsity', type=float, default=1, help='percentage of pixels filled')
p.add_argument('--prior', type=str, default=None, help='prior')
p.add_argument('--downsample', action='store_true', default=False, help='use image downsampling kernel')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, default='custom',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--mask_path', type=str, default=None, help='Path to mask image')
p.add_argument('--custom_image', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\img_set.npy', help='Path to single training image')
p.add_argument('--custom_LEDs', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\LEDs.npy', help='Path to LED location')
p.add_argument('--custom_depth', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\depth.npy', help='Path to LED location')
p.add_argument('--custom_normal', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\normal.npy', help='Path to LED location')
p.add_argument('--custom_mask', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\mask.npy', help='Path to LED location')
p.add_argument('--custom_albedo', type=str, default=r'F:\Project\SIREN\siren\data_rendering\normal_integration\poly2d\ball_albedo_bad_init\albedo.npy', help='Path to LED location')
p.add_argument('--custom_depth_offset', type=float, default=-0.0, help='initial depth from the LED position')
p.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
opt = p.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:{gpu}".format(gpu=opt.gpu_id))


if opt.dataset == 'camera':
    img_dataset = dataio.Camera()
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
    image_resolution = (512, 512)
if opt.dataset == 'camera_downsampled':
    img_dataset = dataio.Camera(downsample_factor=2)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=256, compute_diff='all')
    image_resolution = (256, 256)
if opt.dataset == 'custom':
    img_dataset = dataio.Shading_LEDNPY(opt.custom_image, opt.custom_LEDs, opt.custom_mask, opt.custom_normal, opt.custom_depth, opt.custom_albedo)
    # img_dataset = dataio.SurfaceTent(128)
    if len(img_dataset[0]['img'].shape) == 3:
        numImg, h, w = img_dataset[0]['img'].shape
    else:
        h, w = img_dataset[0]['img'].shape
    image_resolution = (h, w)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, image_resolution, compute_diff='gradients')
    offset = opt.custom_depth_offset

    # image_resolution = (256, 256)

dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, mode='mlp', out_features=1, sidelength=image_resolution, num_hidden_layers = 3,
                                 downsample=opt.downsample, last_layer_offset = offset)

    # model = modules.Siren(in_features=2, out_features=1, hidden_features=256,
    #                       hidden_layers=3, outermost_linear=True)

    # model = modules.SingleBVPNet(type=opt.model_type, in_features=2)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode=opt.model_type, out_features=1, sidelength=image_resolution,
                                 downsample=opt.downsample)
else:
    raise NotImplementedError

# model= torch.nn.DataParallel(model,device_ids = [1])
model.to(device)
# model.cuda()
now = datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
extra_str = 'ball_albedo_bad_init_handleNAN'

root_path = os.path.join(opt.logging_root, opt.experiment_name, '{}_{}'.format(date_time, extra_str))

if opt.custom_mask:
    mask = np.load(opt.custom_mask)
    mask = ToTensor()(mask)
    mask = mask.float().to(device)
    percentage = torch.sum(mask).cpu().numpy() / np.prod(mask.shape)
    print("mask sparsity %f" % (percentage))
else:
    mask = torch.rand(image_resolution) < opt.sparsity
    mask = mask.float().to(device)

# Define the loss
if opt.prior is None:
    loss_fn = partial(loss_functions.render_NL_img_mse_sv_albedo_lstsq, mask.view(-1,1), device = device)
    loss_fn = partial(loss_functions.render_NL_img_mse, mask.view(-1,1))
elif opt.prior == 'TV':
    loss_fn = partial(loss_functions.image_mse_TV_prior, mask.view(-1,1), opt.k1, model)
elif opt.prior == 'FH':
    loss_fn = partial(loss_functions.image_mse_FH_prior, mask.view(-1,1), opt.k1, model)
summary_fn = partial(utils.write_image_summary, image_resolution)


kwargs = {'save_folder': os.path.join(root_path, 'test'),
          'N_gt': np.load(opt.custom_normal),
          'depth_gt': np.load(opt.custom_depth),
          'albedo_gt':np.load(opt.custom_albedo),
          'vmaxND': [10, 1],
          'mask': np.load(opt.custom_mask)}


save_state_path = None #r'F:\Project\SIREN\siren\experiment_scripts\logs\test_inpaint\2021_07_19_15_25_02_pyramid\checkpoints\model_current.pth'
training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, use_lbfgs = False, kwargs = kwargs,
               save_state_path = save_state_path, clip_grad = False, device = device)
