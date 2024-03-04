## Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data

This repository hosts the official PyTorch implementation of the paper: [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data]().


Authored by: Asad Aali, Giannis Daras, Brett Levac, Sidharth Kumar, Alexandros G. Dimakis, Jonathan I. Tamir

<center><img src="https://github.com/asad-aali/ambient-diffusion-mri/blob/main/docs/prior.png" width="512"></center>

<u> Abstract </u>: *We provide a framework for solving inverse problems with diffusion models learned from linearly corrupted data. Our method, Ambient Diffusion Posterior Sampling (A-DPS), leverages a generative model pre-trained on one type of corruption (e.g. image inpainting) to perform posterior sampling conditioned on measurements from a potentially different forward process (e.g. image blurring). We test the efficacy of our approach on standard natural image datasets (CelebA, FFHQ, and AFHQ) and we show that A-DPS can sometimes outperform models trained on clean data for several image restoration tasks in both speed and performance. We further extend the Ambient Diffusion framework to train MRI models with access only to Fourier subsampled multi-coil MRI measurements at various acceleration factors (R = 2, 4, 6, 8). We again observe that models trained on highly subsampled data are better priors for solving inverse problems in the high acceleration regime than models trained on fully sampled data.*

## Installation
The recommended way to run the code is with an Anaconda/Miniconda environment.
First, clone the repository: 

`git clone https://github.com/utcsilab/ambient-diffusion-mri.git`.

Then, create a new Anaconda environment and install the dependencies:

`conda env create -f environment.yml -n ambient`

You will also need to have `diffusers` installed from the source. To do so, run:

`pip install git+https://github.com/huggingface/diffusers.git`

### Download pre-trained models

We provide pre-trained Ambient Diffusion models trained on undersampled FastMRI data, independently at acceleration rates of R = 2, 4, 6, 8. Additionally, we provide a supervised EDM model trained on fully sampled (R = 1) data.
The checkpoints are available through the link: [link](https://utexas.box.com/s/axofnwib9kukdpa92ge4ays87dmuvpf7).
You will need ~1GB of disk space for the models.
To download from the terminal, simply run:

`wget -v -O ambient_models.zip -L https://utexas.box.com/shared/static/axofnwib9kukdpa92ge4ays87dmuvpf7.zip`

### Download dataset

For the experiments, we used a pre-processed version of NYU's FastMRI dataset [here](https://fastmri.med.nyu.edu/). 

To set up the dataset for training/inference, follow the instructions provided [here](https://github.com/NVlabs/edm#preparing-datasets).

## Training New Models

To train a new Ambient Diffusion model on the FastMRI dataset, run the following bash script: `ambient-diffusion-mri/train.sh`

```
R=4
EXPERIMENT_NAME=brainMRI_R=$R
GPUS_PER_NODE=1
GPU=0
DATA_PATH=/path_to_dataset/numpy/ksp_brainMRI_384.zip
OUTDIR=/path_to_output/models/$EXPERIMENT_NAME
CORR=$R
DELTA=5
BATCH=8
METHOD=ambient

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    train.py --gpu=$GPU --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
    --dump=200 --cond=0 --arch=ddpmpp \
    --precond=$METHOD --cres=1,1,1,1 --lr=2e-4 --dropout=0.1 --augment=0 \
    --data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=True \
    --corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
    --normalize=False --fp16=True --wandb_id=$EXPERIMENT_NAME
```

## Sampling

### Generate images from trained model

To generate images from the trained model, run the following bash script: `ambient-diffusion-mri/prior.sh`:

```
R=4
EXPERIMENT_NAME=brainMRI_prior_R=$R
GPUS_PER_NODE=1
GPU=0
MODEL_PATH=/path_to_model/models/brainMRI_R=$R
MAPS_PATH=/path_to_dataset/numpy/ksp_brainMRI_384.zip
SEEDS=1000
BATCH=8

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    prior.py --gpu=$GPU --network=$MODEL_PATH/  --maps_path=$MAPS_PATH\
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEEDS --batch=$BATCH \
    --mask_full_rgb=True --num_masks=1 --guidance_scale=0.0 \
    --training_options_loc=$MODEL_PATH/training_options.json \
    --num=$SEEDS --img_channels=2 --with_wandb=False
```

This will generate 1000 images in the folder `<results/$EXPERIMENT_NAME>`.

### Posterior sampling using Ambient Diffusion Posterior Sampling (A-DPS)

To generate posterior samples given a trained model, run the following bash script: `ambient-diffusion-mri/solve_inverse_adps.sh`:

```
TRAINING_R=4
EXPERIMENT_NAME=brainMRI_ambientDPS
GPUS_PER_NODE=1
GPU=0
MODEL_PATH=/path_to_model/models/brainMRI_R=$TRAINING_R
MEAS_PATH=/path_to_measurements
STEPS=500

for seed in 15
do
    for R in 2 4 6 8
    do
        for sample in {0..100}
        do
            torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
            solve_inverse_adps.py --seed $seed --latent_seeds $seed --gpu $GPU \
            --sample $sample --inference_R $R --training_R $TRAINING_R \
            --l_ss 1 --num_steps $STEPS --S_churn 0 \
            --measurements_path $MEAS_PATH --network $MODEL_PATH \
            --outdir results/$EXPERIMENT_NAME --img_channels 2
        done
    done
done
```

### Posterior sampling using Ambient One-Step (A-OS)

To generate posterior samples given a trained model, run the following bash script: `ambient-diffusion-mri/solve_inverse_1step.sh`:

```
R=4
EXPERIMENT_NAME=brainMRI_1step_R=$R
GPUS_PER_NODE=1
GPU=0
MODEL_PATH=/path_to_model/models/brainMRI_R=$R
MEAS_PATH=/path_to_measurements
SEEDS=100

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    solve_inverse_1step.py --gpu=$GPU --network=$MODEL_PATH/ \
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEEDS --batch=1 \
    --mask_full_rgb=True --training_options_loc=$MODEL_PATH/training_options.json \
    --measurements_path=$MEAS_PATH --num=2 --img_channels=2 --with_wandb=False
```