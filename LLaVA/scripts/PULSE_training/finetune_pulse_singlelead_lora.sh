#!/bin/bash
# LoRA fine-tune PULSE-7B on single-lead ECG images (e.g. Frontier X Plus).
#
# This adapts the existing 12-lead PULSE model to single-lead images. It
# starts FROM the released PULSE-7B checkpoint (not the base LLaVA), so the
# model keeps its ECG knowledge and only adjusts to the single-lead domain.
#
# Sized for a SINGLE GPU (e.g. NVIDIA GB10 / DGX Spark). Scale up
# GPUS_PER_NODE and lower GRAD_ACC_STEP if you have more GPUs.
set -e

# --- distributed (single GPU) ---
export GPUS_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="1234"
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# --- paths (EDIT THESE) ---
model_path=PULSE-ECG/PULSE-7B          # fine-tune from PULSE, not base LLaVA
version=llava_v1

# Produced by: tools/render_single_lead.py build-dataset
data_path=/home/alex/PULSE/data/single_lead/train.json
image_folder=/home/alex/PULSE/data/single_lead/images
output_dir=/home/alex/PULSE/checkpoints/pulse-7b-singlelead-lora

# --- schedule ---
num_epochs=3
BATCH_PER_GPU=4
GLOBAL_BATCH_SIZE=32
TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU))
GRAD_ACC_STEP=$(($GLOBAL_BATCH_SIZE / $TOTAL_BATCH_SIZE))

cd "$(dirname "$0")/../.."   # -> LLaVA/

deepspeed --num_gpus $GPUS_PER_NODE \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model_path \
    --version $version \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_epochs \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
