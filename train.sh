R=2
EXPERIMENT_NAME=brainMRI_R=$R
GPUS_PER_NODE=1
GPU=3
DATA_PATH=/home/asad/ambient-diffusion-mri/data/fastMRI/numpy/ksp_brainMRI_384.zip
OUTDIR=/home/asad/ambient-diffusion-mri/models/$EXPERIMENT_NAME
CORR=$R
DELTA=1
BATCH=8
METHOD=edm

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    train.py --gpu=$GPU --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
    --dump=200 --cond=0 --arch=ddpmpp \
    --precond=$METHOD --cres=1,1,1,1 --lr=2e-4 --dropout=0.1 --augment=0 \
    --data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=True \
    --corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
    --normalize=False --fp16=True --wandb_id=$EXPERIMENT_NAME