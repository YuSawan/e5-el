#!/bin/bash
#SBATCH --job-name=e5el_train
#SBATCH -p gpu_long
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:a6000:1

export WANDB_PROJECT=e5el

configpath=configs/config.yaml
output_dir=save_models/multilingual-e5-large
negative='dense'

for seed in 0 21 42; do
for measure in 'cos' 'ip' 'l2'; do
base_output_dir = ${output_dir}/${seed}/${measure}
mkdir -p ${base_output_dir}

uv run python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file ${configpath} \
    --measure ${measure} \
    --negative ${negative} \
    --seed ${seed} \
    --output_dir ${base_output_dir}/direct \
    --run_name ${base_output_dir}/direct
done
done
