#!/bin/bash

# USAGE: ./scripts/ablation/AugMem_unablated.sh 0 data/core50

GPU="${1:-0}"
OUTDIR="core50_ablation_unablated_outputs"
DATAROOT="${2:-"/media/data/Datasets/Core50"}"

#mkdir -p $OUTDIR/iid/AugMem_ResNet18/
mkdir -p $OUTDIR/class_iid/AugMem_ResNet18/
#mkdir -p $OUTDIR/instance/AugMem_ResNet18/
mkdir -p $OUTDIR/class_instance/AugMem_ResNet18/

python -u experiment_aug.py --scenario class_iid      --n_runs 3   --memory_Nslots 100  --memory_Nfeat 8 --memory_size 200 --lr 0.001   --mem_sparse 5 --replay_coef 5 --replay_times 1 --reg_coef 1000 --logit_coef 1 --n_epoch_first_task 10  --dataroot "$DATAROOT" --freeze_feature_extract --output_dir "$OUTDIR"  --n_epoch 1  --model_type resnet --model_name SqueezeNet --pretrained --agent_type aug_mem --agent_name AugMem  --gpuid "$GPU" --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log    #&
python -u experiment_aug.py --scenario class_instance --n_runs 3   --memory_Nslots 100  --memory_Nfeat 8 --memory_size 200 --lr 0.00025 --mem_sparse 5 --replay_coef 5 --replay_times 1 --reg_coef 1000 --logit_coef 1 --n_epoch_first_task 10  --dataroot "$DATAROOT" --freeze_feature_extract --output_dir "$OUTDIR" --n_epoch 1  --model_type resnet --model_name SqueezeNet --pretrained --agent_type aug_mem --agent_name AugMem  --gpuid "$GPU" --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/AugMem_ResNet18/log.log    #&
