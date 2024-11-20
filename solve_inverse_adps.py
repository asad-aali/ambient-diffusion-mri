import numpy as np
import torch
import os
import argparse
from torch_utils.ambient_diffusion import nrmse_np, psnr, create_masks, nrmse
import pickle
import dnnlib
from torch_utils.misc import StackedRandomGenerator
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
import json
from collections import OrderedDict
import torch
import numpy as np
import sys

class MRI_utils:
    def __init__(self, mask, maps):
        self.mask = mask
        self.maps = maps

    def forward(self,x):
        x_cplx = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[:,None,...]
        coil_imgs = self.maps*x_cplx
        coil_ksp = fft(coil_imgs)
        sampled_ksp = self.mask*coil_ksp
        return sampled_ksp

    def adjoint(self,y):
        sampled_ksp = self.mask*y
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.maps)*coil_imgs,dim=1) #sum over coil dimension

        return img_out[:,None,...]
    
# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    return x

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    return x

def fftmod(x):
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x

def general_forward_SDE_ps(
    y, gt_img,mri_inf_utils, mri_train_utils, corruption_mask, task, l_ss, l_type, net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7, 
    solver='euler', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, verbose = True, training_R=1
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        x_cur = x_cur.requires_grad_() #starting grad tracking with the noised img

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step on Prior.
        h = t_next - t_hat
        
        if training_R == 1:
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        else:
            masked_x_hat = mri_train_utils.adjoint(mri_train_utils.forward(x=x_hat))
            noisy_image = torch.cat((masked_x_hat.real, masked_x_hat.imag), dim=1)
            net_input = torch.cat([noisy_image, torch.ones(1, 2, 384, 320).cuda()], dim=1)
            net_input = torch.cat([noisy_image, corruption_mask*torch.ones(noisy_image.shape[0],corruption_mask.shape[1],corruption_mask.shape[2],corruption_mask.shape[3]).cuda()], dim=1)
            denoised = net(net_input / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)[:, :int(net.img_channels/2)]
        
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised

        # Euler step on liklihood
        if l_type == 'DPS':
            E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
            Ax = mri_inf_utils.forward(x=E_x_start)
        elif l_type == 'ALD':
            Ax = mri_inf_utils.forward(x=denoised)

        residual = y - Ax  
        residual = residual.reshape(latents.shape[0],-1)
        sse_ind = torch.norm(residual,dim=-1)**2
        sse = torch.sum(sse_ind)
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
        x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score

        if task=='mri':
            cplx_recon = mri_transform(x_next) #shape: [B,1,H,W]
            with torch.no_grad():
                nrmse_loss = nrmse(abs(gt_img), abs(cplx_recon))
        if verbose:    
            print('Step:%d , Noise LVL: %.3e,  DC Loss: %.3e,  NRMSE: %.3f'%(i, sigma(t_hat), sse.item(), nrmse_loss.item()))

        # Cleanup 
        x_next = x_next.detach()
        x_cur = x_cur.requires_grad_(False)
    return x_cur

def mri_transform(x):
    return torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[:,None,...] #shape: [1,1,H,W]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=10)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--inference_R', type=int, default=4)
parser.add_argument('--training_R', type=int, default=4)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--latent_seeds', type=int, nargs='+' ,default= [10])
parser.add_argument('--S_churn', type=float, default=40)
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--measurements_path', type=str, default='') 
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='linear') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='none') # ['vp', 'none']
parser.add_argument('--outdir', type=str, default='none') # ['vp', 'none']
parser.add_argument('--network', type=str, default='none') # ['vp', 'none']
parser.add_argument('--img_channels', type=int, default=2) # ['vp', 'none']
parser.add_argument('--method', type=str, default='none') # ['edm', 'ambient']

args   = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')
batch_size=len(args.latent_seeds)

if args.method == 'ambient':
    # load network
    training_options_loc = args.network + "/training_options.json"
    with dnnlib.util.open_url(training_options_loc, verbose=(dist.get_rank() == 0)) as f:
        training_options = json.load(f)
        label_dim = 0

    img_channels = args.img_channels
    if args.training_R == 1:
        interface_kwargs = dict(img_resolution=training_options['dataset_kwargs']['resolution'], label_dim=label_dim, img_channels=img_channels)
    else:
        interface_kwargs = dict(img_resolution=training_options['dataset_kwargs']['resolution'], label_dim=label_dim, img_channels=img_channels*2)
        
    network_kwargs = training_options['network_kwargs']
    model_to_be_initialized = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module

    net_save = args.network + "/network-snapshot.pkl"
    if dist.get_rank() != 0:
            torch.distributed.barrier()
    dist.print0(f'Loading network from "{net_save}"...')
    with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
        loaded_obj = pickle.load(f)['ema']
        modified_dict = OrderedDict({key.replace('_orig_mod.', ''):val for key, val in loaded_obj.items()})
        net = model_to_be_initialized
        net.load_state_dict(modified_dict)

    net = net.to(device)

else:
    # load network
    net_save = args.network + "/network-snapshot.pkl"
    if dist.get_rank() != 0:
            torch.distributed.barrier()
    dist.print0(f'Loading network from "{net_save}"...')
    with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    args.training_R = 1

for args.sample in range(100):
    # designate + create save directory
    args.delta_prob = args.training_R+1
    results_dir = args.outdir + "/trained_r=%d_delta_prob%d/sample%d/seed%d/R=%d"%(args.training_R, args.delta_prob, args.sample, args.seed, args.inference_R)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if os.path.isfile(results_dir + '/checkpoint.pt'):
        print(results_dir + "/checkpoint.pt" + " already exists!")
        continue

    #load data and preprocess
    data_file = args.measurements_path + "/sample_%d.pt"%args.sample
    cont = torch.load(data_file)
    mask_str = 'mask_%d'%args.inference_R

    gt_img = cont['gt'][None,None,...].cuda() #shape [1,1,384,320]
    s_maps = fftmod(cont['s_map'])[None,...].cuda() # shape [1,16,384,320]
    fs_ksp = fftmod(cont['ksp'])[None,...].cuda() #shape [1,16,384,320]
    mask = cont[mask_str][None, ...].cuda() # shape [1,1,384,320]
    ksp = mask*fs_ksp

    # setup MRI forward model + utilities for inferance mask
    mri_inf_utils = MRI_utils(maps=s_maps,mask=mask)
    adj_img = mri_inf_utils.adjoint(ksp)

    # setup MRI forward model + utilities for training mask
    corruption_mask = torch.ones([1, 2, ksp.shape[2], ksp.shape[3]]).cuda()
    args.delta_prob = args.training_R

    if args.training_R > 1:
        args.delta_prob = args.training_R+1
        corruption_mask[:,0] = create_masks(args.training_R, args.delta_prob, 20, ksp.shape[-2], ksp.shape[-1])[None]

    mri_train_utils = MRI_utils(maps=s_maps, mask=corruption_mask[:,0])

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    # Pick latents and labels.
    rnd = StackedRandomGenerator(device, args.latent_seeds)
    latents = rnd.randn([batch_size, 2, gt_img.shape[-2], gt_img.shape[-1]], device=device)
    class_labels = None

    image_recon = general_forward_SDE_ps(y=ksp,  gt_img=gt_img, mri_inf_utils=mri_inf_utils, mri_train_utils=mri_train_utils, corruption_mask=corruption_mask, task='mri', l_type='ALD', l_ss=args.l_ss, 
        net=net, latents=latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=args.num_steps, sigma_min=0.004, sigma_max=args.sigma_max, rho=7,
        solver=args.solver, discretization=args.discretization, schedule='linear', scaling=args.scaling,
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        S_churn=args.S_churn, S_min=0, S_max=float('inf'), S_noise=1, verbose = True, training_R = args.training_R)

    cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[:,None] #shape: [1,1,H,W]

    cplx_recon=cplx_recon.detach().cpu().numpy()
    mean_recon=np.mean(cplx_recon,axis=0)[None]
    gt_img=gt_img.cpu().numpy()
    img_nrmse = nrmse_np(abs(gt_img[0,0]), abs(mean_recon[0,0]))
    img_SSIM = ssim(abs(gt_img[0,0]), abs(mean_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())
    img_PSNR = psnr(gt=abs(gt_img[0,0]), est=abs(mean_recon[0]),max_pixel=np.amax(abs(gt_img)))

    # print('cplx net out shape: ',cplx_recon.shape)
    print('Sample %d, seed %d, R: %d, NRMSE: %.3f, SSIM: %.3f, PSNR: %.3f'%(args.sample, args.seed, args.inference_R, img_nrmse, img_SSIM, img_PSNR))

    dict = { 
            'gt_img': gt_img,
            'recon': cplx_recon,
            'adj_img': adj_img.cpu().numpy(),
            'nrmse': img_nrmse,
            'ssim': img_SSIM,
            'psnr': img_PSNR
    }

    torch.save(dict, results_dir + '/checkpoint.pt')