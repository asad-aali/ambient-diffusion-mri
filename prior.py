import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from dnnlib.util import print_tensor_stats, tensor_clipping, save_images
from torch_utils import distributed as dist
from training import dataset
import scipy.linalg
import wandb
from torch_utils.ambient_diffusion import get_random_mask
from torch_utils.misc import parse_int_list
from torch_utils.misc import StackedRandomGenerator
import time
import random
import json
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import sigpy as sp
import zipfile

def cdist_masked(x1, x2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(x1)
        mask2 = torch.ones_like(x2)
    x1 = x1[0].unsqueeze(0)
    diffs = x1.unsqueeze(1) - x2.unsqueeze(0)
    combined_mask = mask1.unsqueeze(1) * mask2.unsqueeze(0)
    error = 0.5 * torch.linalg.norm(combined_mask * diffs)**2
    return error

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

def create_masks(R, delta_R, acs_lines, LENGTH_Y, LENGTH_X):
    if delta_R == 3:
        delta_R=54
    elif delta_R == 5:
        delta_R=16
    elif delta_R == 7:
        delta_R=8
    elif delta_R == 9:
        delta_R=5

    total_lines = LENGTH_X
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    
    random.shuffle(random_line_idx)
    further_mask = mask.copy()
    further_mask[:, random_line_idx[0:delta_R]] = 0.

    mask = sp.resize(mask, [LENGTH_Y, LENGTH_X])
    further_mask = sp.resize(further_mask, [LENGTH_Y, LENGTH_X])
    diff_dim = LENGTH_Y - LENGTH_X
    half_dim = int(diff_dim / 2)
    mask[0:half_dim] = mask[half_dim:diff_dim]
    mask[LENGTH_Y-half_dim:LENGTH_Y] = mask[half_dim:diff_dim]
    further_mask[0:half_dim] = further_mask[half_dim:diff_dim]
    further_mask[LENGTH_Y-half_dim:LENGTH_Y] = further_mask[half_dim:diff_dim]

    return torch.tensor(further_mask)

def ambient_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampler_seed=42, survival_probability=0.54,
    mask_full_rgb=False,
    same_for_all_batch=False,
    num_masks=1,
    guidance_scale=0.0,
    clipping=True,
    static=False,  # whether to use soft clipping or static clipping
    resample_guidance_masks=False,
    experiment_name=None,
    maps_path=None
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    latents = latents[:,:,:,0:320]
    corr = int(experiment_name[-1])
    delta_corr = corr + 1
    
    file = zipfile.ZipFile(maps_path, 'r')
    masks = torch.ones_like(latents).cuda()
    maps = torch.zeros([latents.shape[0], 4, latents.shape[-2], latents.shape[-1]], dtype=torch.complex64).cuda()

    for i in range(latents.shape[0]):
        with file.open(str(i) + "/maps.npy", 'r') as f:
            map = np.load(f)
            map = fftmod(map)
            map = torch.tensor(map, dtype=torch.complex64).to(latents.device)
            maps[i] = map
        mask_delta = create_masks(corr, delta_corr, 20, latents.shape[-2], latents.shape[-1])
        masks[i,0] = torch.tensor(mask_delta)

    clean_image = None

    print('Maps Shape: ' + str(maps.shape) + '\n')
    print('Masks Shape: ' + str(masks.shape) + '\n')
    print('Image Shape: ' + str(latents.shape) + '\n')
    print("Mask R=" + str(delta_corr) + ": " + str(float((latents.shape[-2]*latents.shape[-1])/torch.sum(mask_delta))))
    masks = masks[None].cuda()

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        x_hat = x_hat.detach()
        x_hat.requires_grad = True

        denoised = []
        for mask_index in range(num_masks):
            corruption_mask = masks[mask_index]
            x_hat_cplx = x_hat[:,0] + 1j*x_hat[:,1]
            x_hat_cplx = x_hat_cplx[:,None,...]
            masked_x_hat = adjoint(forward(x_hat_cplx, maps, corruption_mask[:,0][:,None]), maps, corruption_mask[:,0][:,None])
            masked_x_hat = torch.cat((masked_x_hat.real, masked_x_hat.imag), dim=1)

            noisy_image = masked_x_hat

            net_input = torch.cat([noisy_image, corruption_mask], dim=1)
            net_output = net(net_input, t_hat, class_labels).to(torch.float64)[:, :int(net.img_channels/2)]

            if clipping:
                net_output = tensor_clipping(net_output, static=static)

            if clean_image is not None:
                net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image

            # Euler step.
            denoised.append(net_output)


        stack_denoised = torch.stack(denoised)
        flattened = stack_denoised.view(stack_denoised.shape[0], -1)
        l2_norm = cdist_masked(flattened, flattened, None, None)
        l2_norm = l2_norm.mean()
        rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]

        clean_pred = stack_denoised[0]

        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        grad_1 = single_mask_grad - guidance_scale * rec_grad
        x_next += grad_1

        if i < num_steps - 1:
            x_next = x_next.detach()
            x_next.requires_grad = True

            denoised = []
            for mask_index in range(num_masks):
                corruption_mask = masks[mask_index]

                x_next_cplx = x_next[:,0] + 1j*x_next[:,1]
                x_next_cplx = x_next_cplx[:,None,...]
                masked_image = adjoint(forward(x_next_cplx, maps, corruption_mask[:,0][:,None]), maps, corruption_mask[:,0][:,None])
                masked_image = torch.cat((masked_image.real, masked_image.imag), dim=1)
                
                # masked_image = corruption_mask * x_next
                noisy_image = masked_image
                net_input = torch.cat([noisy_image, corruption_mask], dim=1)
                net_output = net(net_input, t_next, class_labels).to(torch.float64)[:, :int(net.img_channels/2)]
                if clipping:
                    net_output = tensor_clipping(net_output, static=static)
                
                if clean_image is not None:
                    net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image
                denoised.append(net_output)
            
            stack_denoised = torch.stack(denoised)
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None)
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_next)[0]
            clean_pred = stack_denoised[0]
            single_mask_grad = (t_next - t_hat) * (x_next - clean_pred) / t_next
            grad_2 = single_mask_grad - guidance_scale * rec_grad
            x_next = x_hat + 0.5 * (grad_1 + grad_2)
        else:
            if clean_image is not None:
                x_next = masks[0] * x_next + (1 - masks[0]) * clean_image
            else:
                clean_image = x_next
                x_next = x_hat + grad_1
    return x_next



@click.command()
@click.option('--with_wandb', help='Whether to report to wandb', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--network', 'network_loc',  help='Location of the folder where the network is stored', metavar='PATH|URL',                      type=str, required=True)
@click.option('--maps_path',  help='Location of the folder where the maps are stored', metavar='PATH|URL',                      type=str, required=True)
@click.option('--training_options_loc', help='Location of the training options file', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds', metavar='INT',                      type=int, default='0', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--gpu',                     help='GPU Machine', metavar='INT',                                       type=int, default=0, show_default=True)

@click.option('--img_channels', help='Channels for image', metavar='INT', type=int, default=3, show_default=True)
@click.option('--corruption_probability', help='Probability of corruption', metavar='FLOAT', type=float, default=0.4, show_default=True)
@click.option('--delta_probability', help='Probability of delta corruption', metavar='FLOAT', type=float, default=0.1, show_default=True)

@click.option('--num_masks', help='Number of sampling masks', default=1, show_default=True, type=int)
@click.option('--guidance_scale', help='How much to rely on scaling', default=0.0, show_default=True, type=float)

@click.option('--mask_full_rgb', help='Whether to mask the full RGB channel.', default=False, show_default=True, required=True)


@click.option('--experiment_name', help="Name of the experiment to log to wandb", type=str, required=True)
@click.option('--wandb_id', help='Id of wandb run to resume', type=str, default='')
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--eval_step', help='Number of steps between evaluations', metavar='INT', type=int, default=1, show_default=True)
@click.option('--skip_generation', help='Skip image generation and only compute metrics', default=False, required=False, type=bool)
@click.option('--skip_calculation', help='Skip metrics', default=False, required=False, type=bool)

# if the network is class conditional, the number of classes it is trained on must be specified
@click.option('--num_classes',             help='Number of classes', metavar='INT', type=int, default=0, show_default=True)

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
         num_masks, guidance_scale, mask_full_rgb,
         # other params
         experiment_name, wandb_id, ref_path, num_expected, seed, eval_step, skip_generation, gpu,
         skip_calculation, num_classes, maps_path,
         device=torch.device('cuda'),  **sampler_kwargs):
    torch.multiprocessing.set_start_method('spawn')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dist.init()

    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    seeds = range(seeds)
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

                    # Generate images.
                    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                    images = ambient_sampler(net, latents, class_labels, randn_like=rnd.randn_like, sampler_seed=batch_seeds, 
                        num_masks=num_masks, guidance_scale=guidance_scale, maps_path=maps_path,
                        mask_full_rgb=mask_full_rgb, experiment_name=experiment_name, **sampler_kwargs)

                    curr_seed = batch_seeds[0]
                    image_dir = os.path.join(outdir, str(checkpoint_number), 
                                            f'collage-{curr_seed-curr_seed%1000:06d}') if subdirs else os.path.join(outdir, str(checkpoint_number), "collages")
                    dist.print0(f"Saving loc: {image_dir}")
                    image_path = os.path.join(image_dir, f'collage-{curr_seed:06d}.png')

                    if img_channels == 2:
                        images_np = images[:,0,...].cpu().detach() + 1j*images[:,1,...].cpu().detach()
                        for seed, image_np in zip(batch_seeds, images_np):
                            image_dir = os.path.join(outdir, str(checkpoint_number), f'{seed-seed%1000:06d}') if subdirs else os.path.join(outdir, str(checkpoint_number))
                            os.makedirs(image_dir, exist_ok=True)
                            image_path = os.path.join(image_dir, f'{seed:06d}.png')
                            plt.figure(frameon=False)
                            plt.imshow(torch.flipud(torch.abs(image_np)), cmap='gray')
                            plt.axis('off')
                            plt.savefig(image_path, transparent=True, bbox_inches='tight', pad_inches=0)
                            plt.close()
                    else:
                        # Save images.
                        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                        for seed, image_np in zip(batch_seeds, images_np):
                            image_dir = os.path.join(outdir, str(checkpoint_number), f'{seed-seed%1000:06d}') if subdirs else os.path.join(outdir, str(checkpoint_number))
                            os.makedirs(image_dir, exist_ok=True)
                            image_path = os.path.join(image_dir, f'{seed:06d}.png')
                            if image_np.shape[2] == 1:
                                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                            else:
                                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                    batch_count += 1
                    
                dist.print0(f"Node finished generation for {checkpoint_number}")
                dist.print0("waiting for others to finish..")

            # Rank 0 goes first.
            if dist.get_rank() != 0:
                torch.distributed.barrier()
            dist.print0("Everyone finished.. Starting calculation..")

if __name__ == "__main__":
    main()