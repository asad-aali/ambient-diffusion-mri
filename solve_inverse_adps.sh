TRAINING_R=1
EXPERIMENT_NAME=brainMRI_ambientDPS
GPUS_PER_NODE=1
GPU=0
MODEL_PATH=/home/asad/ambient-diffusion-mri/models/brainMRI_R=$TRAINING_R
MEAS_PATH=/csiNAS/asad/data/brain_fastMRI/val_samples_ambient
STEPS=500
METHOD=edm

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
            --outdir results/$EXPERIMENT_NAME --img_channels 2 --method $METHOD
        done
    done
done