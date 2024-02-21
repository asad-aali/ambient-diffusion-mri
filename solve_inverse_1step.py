import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training import dataset
import scipy.linalg
import wandb
from torch_utils.misc import parse_int_list
from torch_utils.misc import StackedRandomGenerator
import time
import json
from collections import OrderedDict
import warnings
from training.dataset import ImageFolderDataset
from torch_utils import misc
import matplotlib.pyplot as plt
import sys

def fftmod(x):
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x

# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    return x

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    return x

def forward(image, maps, mask):
    coil_imgs = maps*image
    coil_ksp = fft(coil_imgs)
    sampled_ksp = mask*coil_ksp
    return sampled_ksp

def adjoint(ksp, maps, mask):
    sampled_ksp = mask*ksp
    coil_imgs = ifft(sampled_ksp)
    img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1)[:,None,...] #sum over coil dimension

    return img_out

def ambient_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampler_seed=42,
    mask_full_rgb=False,
    same_for_all_batch=False,
    clipping=True,
    static=True,  # whether to use soft clipping or static clipping'
    measurements_path=None,
    experiment_name=None,
):
    contents = torch.load(measurements_path)

    print('\nForwardFastMRI Dataloader:\n')
    maps     = fftmod(contents['s_map'])[None].cuda() # shape: [1,C,H,W]
    mask     = contents['mask_' + str(experiment_name[-1])][None].cuda() # shape: [1,1,H,W]
    gt_img   = contents['gt'][None,None].cuda() # shape [1,1,H,W]
    ksp      = forward(gt_img, maps, mask)
    latents  = latents[:,:, 0:ksp.shape[2], 0:ksp.shape[3]]

    print('Ksp Shape: ' + str(ksp.shape) + '\n')
    print('Maps Shape: ' + str(maps.shape) + '\n')
    print('Mask Shape: ' + str(mask.shape) + '\n')
    print('Image Shape: ' + str(gt_img.shape) + '\n')

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    corruption_mask = torch.ones([1, 2, ksp.shape[2], ksp.shape[3]]).to(device=latents.device)

    corruption_mask[:,0] = mask[0]
    print("\nMask R=" + str(experiment_name[-1]) + ": " + str(float((latents.shape[-2]*latents.shape[-1])/torch.sum(corruption_mask[:,0][0]))))

    ksp_adjoint = adjoint(ksp, maps, mask)

    noisy_image = torch.cat((ksp_adjoint.real, ksp_adjoint.imag), dim=1)
    net_input = torch.cat([noisy_image, corruption_mask], dim=1)
    net_output = net(net_input, torch.zeros(net_input.shape[0]).cuda() + 0.0030, class_labels).to(torch.float64)[:, :int(net.img_channels/2)]

    net_output_cplx = net_output[:,0] + 1j*net_output[:,1]
    nrmse = torch.linalg.norm(net_output_cplx - gt_img) / torch.linalg.norm(gt_img)
    print('\n\nNRMSE: ' + str(float(nrmse)))
    return net_output, ksp, gt_img


@click.command()
@click.option('--with_wandb', help='Whether to report to wandb', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--network', 'network_loc',  help='Location of the folder where the network is stored', metavar='PATH|URL',                      type=str, required=True)
@click.option('--training_options_loc', help='Location of the training options file', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--gpu',                     help='GPU Machine', metavar='INT',                                       type=int, default=0, show_default=True)

@click.option('--img_channels', help='Channels for image', metavar='INT', type=int, default=3, show_default=True)
@click.option('--corruption_probability', help='Probability of corruption', metavar='FLOAT', type=float, default=0.4, show_default=True)
@click.option('--delta_probability', help='Probability of delta corruption', metavar='FLOAT', type=float, default=0.1, show_default=True)

@click.option('--mask_full_rgb', help='Whether to mask the full RGB channel.', default=False, show_default=True, required=True)


@click.option('--experiment_name', help="Name of the experiment to log to wandb", type=str, required=True)
@click.option('--wandb_id', help='Id of wandb run to resume', type=str, default='')
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--eval_step', help='Number of steps between evaluations', metavar='INT', type=int, default=1, show_default=True)
@click.option('--skip_generation', help='Skip image generation and only compute metrics', default=False, required=False, type=bool)
@click.option('--skip_calculation', help='Skip metrics', default=True, required=False, type=bool)

# if the network is class conditional, the number of classes it is trained on must be specified
@click.option('--num_classes',             help='Number of classes', metavar='INT', type=int, default=0, show_default=True)

# Forward Operator params
@click.option('--corruption_pattern',     help='Corruption pattern', metavar='dust|box_masking|averaging|blurring|compressed_sensing', 
    type=click.Choice(['box_masking', 'dust', 'averaging', 'blurring', 'compressed_sensing']), default='averaging', show_default=True)

# masking
@click.option('--operator_corruption_probability', help='Probability of corruption', metavar='FLOAT', type=float, default=0.4, show_default=True)

# downsampling
@click.option('--downsampling_factor',    help='Downsampling factor', metavar='INT', type=int, default=8, show_default=True)


# compressed sensing
@click.option('--num_measurements',      help='Number of measurements', metavar='INT', type=int, default=32, show_default=True)

# blurring
@click.option('--blur_type',    help='Blurring type', metavar='motion|gaussian', type=click.Choice(['motion', 'gaussian']), default='motion', show_default=True)
@click.option('--kernel_size',   help='Kernel size', metavar='INT', type=int, default=31, show_default=True)
@click.option('--kernel_std',   help='Kernel std', metavar='FLOAT', type=float, default=3, show_default=True)




# Measurements
@click.option('--measurements_path',      help='Path to the measurements', metavar='PATH', type=str, required=True)


@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(with_wandb, network_loc, training_options_loc, outdir, subdirs, seeds, class_idx, max_batch_size, 
         # Ambient Diffusion Params
         img_channels, corruption_probability, delta_probability,
         mask_full_rgb,
         # other params
         experiment_name, wandb_id, ref_path, num_expected, seed, eval_step, skip_generation,
         skip_calculation, num_classes, corruption_pattern, operator_corruption_probability, 
         downsampling_factor, num_measurements, 
         blur_type, kernel_size, kernel_std, 
         measurements_path, gpu,
         device=torch.device('cuda'),  **sampler_kwargs):
    torch.multiprocessing.set_start_method('spawn')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dist.init()
    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    seeds = seeds[:num_expected]
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()

    dist.print0(f"The algorithm will run for {num_batches} batches --  {len(seeds)} images of batch size {max_batch_size}")
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # the following has for each batch size allocated to this GPU, the indexes of the corresponding images.
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    batches_per_process = len(rank_batches)
    dist.print0(f"This process will get {len(rank_batches)} batches.")

    if dist.get_rank() == 0 and with_wandb:
        wandb.init(
            project="ambient_diffusion",
            name=experiment_name,
            id=wandb_id if wandb_id else None,
            resume="must" if wandb_id else False
        )
        dist.print0("Initialized wandb")

    if not skip_generation:
        # load training options
        with dnnlib.util.open_url(training_options_loc, verbose=(dist.get_rank() == 0)) as f:
            training_options = json.load(f)

        if training_options['dataset_kwargs']['use_labels']:
            assert num_classes > 0, "If the network is class conditional, the number of classes must be positive."
            label_dim = num_classes
        else:
            label_dim = 0
        interface_kwargs = dict(img_resolution=training_options['dataset_kwargs']['resolution'], label_dim=label_dim, img_channels=img_channels*2)
        network_kwargs = training_options['network_kwargs']
        model_to_be_initialized = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module

        eval_index = 0  # keeps track of how many checkpoints we have evaluated so far
        while True:
            # find all *.pkl files in the folder network_loc and sort them
            files = dnnlib.util.list_dir(network_loc)
            # Filter the list to include only "*.pkl" files
            pkl_files = [f for f in files if f.endswith('.pkl')]
            # Sort the list of "*.pkl" files
            sorted_pkl_files = sorted(pkl_files)[eval_index:]

            if len(sorted_pkl_files) == 0:
                dist.print0("No new checkpoint found! Going to sleep for 1min!")
                time.sleep(60)
                dist.print0("Woke up!")
            
            for checkpoint_number in zip(sorted_pkl_files):
                # Rank 0 goes first.
                if dist.get_rank() != 0:
                    torch.distributed.barrier()

                network_pkl = os.path.join(network_loc, f'network-snapshot.pkl')
                # Load network.
                dist.print0(f'Loading network from "{network_pkl}"...')
                with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
                    loaded_obj = pickle.load(f)['ema']
                
                if type(loaded_obj) == OrderedDict:
                    COMPILE = False
                    if COMPILE:
                        net = torch.compile(model_to_be_initialized)
                        net.load_state_dict(loaded_obj)
                    else:
                        modified_dict = OrderedDict({key.replace('_orig_mod.', ''):val for key, val in loaded_obj.items()})
                        net = model_to_be_initialized
                        net.load_state_dict(modified_dict)
                else:
                    # ensures backward compatibility for times where net is a model pkl file
                    net = loaded_obj
                net = net.to(device)
                dist.print0(f'Network loaded!')

                # Other ranks follow.
                if dist.get_rank() == 0:
                    torch.distributed.barrier()

                # Loop over batches.
                dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
                batch_count = 1
                for batch_seeds in tqdm.tqdm(rank_batches, disable=dist.get_rank() != 0):
                    dist.print0(f"Waiting for the green light to start generation for {batch_count}/{batches_per_process}")
                    # don't move to the next batch until all nodes have finished their current batch
                    torch.distributed.barrier()
                    dist.print0("Others finished. Good to go!")
                    batch_size = len(batch_seeds)
                    if batch_size == 0:
                        continue

                    # Pick latents and labels.
                    rnd = StackedRandomGenerator(device, batch_seeds)
                    latents = rnd.randn([batch_size, int(net.img_channels/2), net.img_resolution, net.img_resolution], device=device)
                    class_labels = None
                    if net.label_dim:
                        class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
                    if class_idx is not None:
                        class_labels[:, :] = 0
                        class_labels[:, class_idx] = 1

                    # load images from dataset
                    curr_seed = batch_seeds[0]
                    os.makedirs(os.path.join(outdir), exist_ok=True)

                    for curr_seed in range(int(seeds[0])):
                        # Generate images.
                        measurements_path_temp = measurements_path + "/sample_" + str(curr_seed) + ".pt"
                        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                        images, ksp, gt = ambient_sampler(net, latents, class_labels, randn_like=rnd.randn_like, sampler_seed=batch_seeds,
                                                          mask_full_rgb=mask_full_rgb, measurements_path=measurements_path_temp, 
                                                          experiment_name=experiment_name, **sampler_kwargs)

                        image_dir = outdir

                        dist.print0(f"Saving loc: {image_dir}")
                        image_path = os.path.join(image_dir, f'collage-{curr_seed:06d}.png')
                        images_np = images[:,0,...].cpu().detach() + 1j*images[:,1,...].cpu().detach()
                        for seed, image_np in zip(batch_seeds, images_np):
                            image_dir = os.path.join(outdir, str(checkpoint_number), f'{seed-curr_seed%1000:06d}') if subdirs else os.path.join(outdir, str(checkpoint_number))
                            os.makedirs(image_dir, exist_ok=True)
                            image_path = os.path.join(image_dir, f'{curr_seed:06d}.png')
                            plt.figure(frameon=False)
                            plt.imshow(torch.flipud(torch.abs(image_np)), cmap='gray', vmax=1)
                            plt.axis('off')
                            plt.savefig(image_path, transparent=True, bbox_inches='tight', pad_inches=0)
                            plt.close()
                            torch.save({'recon': image_np}, os.path.join(image_dir, f'{curr_seed:06d}.pt'))

                            plt.figure(frameon=False)
                            plt.imshow(torch.flipud(torch.abs(ksp[0,0].cpu().detach())), cmap='gray', vmax=1)
                            plt.axis('off')
                            plt.savefig(os.path.join(image_dir, f'{curr_seed:06d}_adj.png'), transparent=True, bbox_inches='tight', pad_inches=0)
                            plt.close()

                            plt.figure(frameon=False)
                            plt.imshow(torch.flipud(torch.abs(gt[0,0].cpu().detach())), cmap='gray', vmax=1)
                            plt.axis('off')
                            plt.savefig(os.path.join(image_dir, f'{curr_seed:06d}_gt.png'), transparent=True, bbox_inches='tight', pad_inches=0)
                            plt.close()
                    
                dist.print0(f"Node finished generation for {checkpoint_number}")
                dist.print0("waiting for others to finish..")

if __name__ == "__main__":
    main()