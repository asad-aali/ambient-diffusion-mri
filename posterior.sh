EXPERIMENT_NAME=brainMRI_384_posterior_5
GPUS_PER_NODE=1
MODEL_PATH=/home/asad/ambient-diffusion-mri/models/brainMRI_ksp_384/00000-ksp_brainMRI_384-uncond-ddpmpp-ambient-gpus4-batch32-fp16
MEAS_PATH=/csiNAS/asad/data/brain_fastMRI/val_samples_whitened
CORR_PROB=4
DELTA_PROB=5
STEPS=300
SEED=0
SAMPLE_FILE=sample_0_R=5

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    solve_inverse_problems.py --network=$MODEL_PATH/ \
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEED --batch=1 \
    --corruption_probability=$CORR_PROB --delta_probability=$DELTA_PROB --mask_full_rgb=True \
    --training_options_loc=$MODEL_PATH/training_options.json \
    --measurements_path=$MEAS_PATH/$SAMPLE_FILE.pt \
    --num=2 --img_channels=2 --with_wandb=False --steps=$STEPS