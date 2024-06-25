#!/bin/bash
deepspeed --num_gpus 8 /home1/rsethi1/stigmatizing_lang_rsh/LLaMA-Factory/src/train.py \
    --deepspeed /home1/rsethi1/stigmatizing_lang_rsh/inputs/configs/ds_config.json \
    --stage dpo \
    --do_train \
    --model_name_or_path /home1/rsethi1/stigmatizing_lang_rsh/outputs/models/cai_sft \
    --dataset _ \
    --template llama3 \
    --output_dir /home1/rsethi1/stigmatizing_lang_rsh/outputs/models/cai_dpo \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 0.000005 \
    --num_train_epochs 3.0 \
    --fp16 \
    --finetuning_type lora \
    --lora_target all \
    --pref_beta 0.1 \
    --pref_loss sigmoid
    --plot_loss