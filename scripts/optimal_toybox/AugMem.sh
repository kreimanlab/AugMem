# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
# ./scripts/AugMem_toybox.sh toybox 1

DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p $OUTDIR/iid/AugMem_ResNet18/
mkdir -p $OUTDIR/class_iid/AugMem_ResNet18/
#mkdir -p $OUTDIR/instance/AugMem_ResNet18/
mkdir -p $OUTDIR/class_instance/AugMem_ResNet18/



#--visualize
python -u experiment_aug.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT --replay_coef 5 --output_dir $OUTDIR  --logit_coef 1 --first_times 1 --freeze_feature_extract --replay_times 1 --reg_coef 1000 --n_epoch 1  --memory_size 200 --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 500  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 20 --n_workers 20 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log    #&
# TODO: change above to n_runs 10
#--freeze_feature_extract
#--visualize
python -u experiment_aug.py --scenario class_instance --freeze_feature_extract --dataset $DATASET --dataroot $DATAROOT --lr 0.0001 --logit_coef 1 --first_times 1 --output_dir $OUTDIR --replay_coef 5 --replay_times 1 --memory_size 200 --reg_coef 1000 --n_epoch 1 --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 500  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 20 --n_workers 8 | tee ${OUTDIR}/class_instance/AugMem_ResNet18/log.log   #&

########### normal #########

#python -u experiment_aug.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT --replay_coef 5 --output_dir $OUTDIR  --first_times 1 --replay_times 1 --reg_coef 1000 --n_epoch 1  --memory_size 200 --freeze_feature_extract --n_runs 1 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log    #&

#python -u experiment_aug.py --scenario class_instance --dataset $DATASET --dataroot $DATAROOT --freeze_feature_extract --lr 0.00001 --first_times 3 --output_dir $OUTDIR --replay_coef 5 --replay_times 1 --memory_size 200 --reg_coef 1000 --n_epoch 1 --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_instance/AugMem_ResNet18/log.log    #&
