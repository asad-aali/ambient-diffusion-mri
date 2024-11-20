TRAINING_R=6
GPUS_PER_NODE=1
GPU=2
MODEL_PATH=/home/asad/ambient-diffusion-mri/models/brainMRI_R=$TRAINING_R
MEAS_PATH=/csiNAS/asad/DATA-FastMRI/brain/val/32dB
STEPS=500
METHOD=ambient
EXPERIMENT_NAME=AmbientALD

for seed in {15..18}
do
    for R in 2 4 6 8
    do
        python solve_inverse_adps.py --seed $seed --latent_seeds $seed --gpu $GPU \
        --sample 0 --inference_R $R --training_R $TRAINING_R \
        --l_ss 1 --num_steps $STEPS --S_churn 0 \
        --measurements_path $MEAS_PATH --network $MODEL_PATH \
        --outdir results/$EXPERIMENT_NAME --img_channels 2 --method $METHOD
    done
done