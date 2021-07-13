# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='test_depth', required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--depth_map_path', type=str, default='F:/Project/SIREN/siren/data_rendering/depth_approx/lena_r.npy',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--mask_path', type=str, default='F:/Project/SIREN/siren/data_rendering/depth_approx/mask.npy',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()


import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
# curr_gpuid = torch.cuda.current_device()
# print(curr_gpuid)

depth_dataset = dataio.DepthMap(opt.depth_map_path, None)
# coord_dataset = dataio.Implicit2DWrapper(depth_dataset, sidelength=512, compute_diff='all')
dataloader = DataLoader(depth_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)


# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=2)
model.cuda()


# Define the loss
loss_fn = loss_functions.depth_approx
summary_fn = utils.write_depth_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
