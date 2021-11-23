#!/usr/bin/env bash

PROJ_ROOT=/home/mengmi/Projects/Proj_CL_NTM/cvpr22/Remind

export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}/image_classification_experiments

IMAGE_DIR=/home/mengmi/Projects/Proj_CL_NTM/cvpr22/cifar100
EXPT_NAME=cifar100
SCENARIO=class_iid
BASE_INIT_CLASSES=5
CLASS_INCREMENT=5
NUM_CLASSES=100
GPU=0


REPLAY_SAMPLES=50
MAX_BUFFER_SIZE=959665
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CKPT=./imagenet_files/best_ResNet18ClassifyAfterLayer4_1_100.pth # base init ckpt file
LABEL_ORDER_DIR=./imagenet_files/ # location of numpy label files
SAVE_DIR=./results/

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_experiment.py \
--runs 0 \
--scenario ${SCENARIO} \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier ResNet18_StartAt_Layer4_1 \
--classifier_ckpt ${BASE_INIT_CKPT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.1 \
--end_lr 0.001 \
--lr_step_size 100 \
--lr_mode step_lr_per_class \
--weight_decay 1e-5 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--save_dir ${SAVE_DIR} \
--expt_name ${EXPT_NAME} 
