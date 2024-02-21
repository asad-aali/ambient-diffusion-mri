# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, **kwargs):
        images = images[:,:,:,32:352]
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


#----------------------------------------------------------------------------
# EDMLoss for Ambient Diffusion

@persistence.persistent_class
class AmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm

    # Centered, orthogonal fft in torch >= 1.7
    def fft(self, x):
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        return x

    # Centered, orthogonal ifft in torch >= 1.7
    def ifft(self, x):
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        return x
    
    def forward(self, image, maps, mask):
        coil_imgs = maps*image
        coil_ksp = self.fft(coil_imgs)
        sampled_ksp = mask*coil_ksp
        return sampled_ksp

    def adjoint(self, ksp, maps, mask):
        sampled_ksp = mask*ksp
        coil_imgs = self.ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1)[:,None,...] #sum over coil dimension
        return img_out

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None):        
        images = images[:,:,:,32:352]
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        y_noisy = y + n
        y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
        y_noisy_cplx = y_noisy_cplx[:,None,...]
        
        noisy_image = self.adjoint(self.forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
        
        hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
        hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

        cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        D_yn_cplx = D_yn[:,0] + 1j*D_yn[:,1]
        D_yn_cplx = D_yn_cplx[:,None,...]
        masked_D_yn = self.adjoint(self.forward(D_yn_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_D_yn = torch.cat((masked_D_yn.real, masked_D_yn.imag), dim=1)
        masked_D_yn_hat = self.adjoint(self.forward(D_yn_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_D_yn_hat = torch.cat((masked_D_yn_hat.real, masked_D_yn_hat.imag), dim=1)

        y_cplx = y[:,0] + 1j*y[:,1]
        y_cplx = y_cplx[:,None,...]
        masked_y = self.adjoint(self.forward(y_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_y = torch.cat((masked_y.real, masked_y.imag), dim=1)
        masked_y_hat = self.adjoint(self.forward(y_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_y_hat = torch.cat((masked_y_hat.real, masked_y_hat.imag), dim=1)
        
        if self.norm == 2:
            train_loss = weight * ((masked_D_yn_hat - masked_y_hat) ** 2)
            val_loss = weight * ((masked_D_yn - masked_y) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss
#----------------------------------------------------------------------------
# VPLoss for Ambient Diffusion
@persistence.persistent_class
class AmbientVPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, norm=2):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.norm = norm

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        cat_input = torch.cat([hat_corruption_matrix * (y + n), hat_corruption_matrix], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        if self.norm == 2:
            train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
            val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss


    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
