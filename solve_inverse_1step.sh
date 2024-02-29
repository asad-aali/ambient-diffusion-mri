R=2
EXPERIMENT_NAME=brainMRI_1step_R=$R
GPUS_PER_NODE=1
GPU=1
MODEL_PATH=/home/asad/ambient-diffusion-mri/models/brainMRI_R=$R
MEAS_PATH=/csiNAS/asad/data/brain_fastMRI/val_samples_ambient
SEEDS=100

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    solve_inverse_1step.py --gpu=$GPU --network=$MODEL_PATH/ \
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEEDS --batch=1 \
    --mask_full_rgb=True --training_options_loc=$MODEL_PATH/training_options.json \
    --measurements_path=$MEAS_PATH --num=2 --img_channels=2 --with_wandb=False