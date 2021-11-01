# Param #1: dataset name, e.g. core50, toybox, ilab2mlight, cifar100. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
    #DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/ilab/iLab-2M-Light/"
elif [ "$DATASET" = "cifar100" ]; then
    DATAROOT="/media/data/morgan_data/cifar100"
else
    echo "Invalid dataset name!"
    exit
fi

mkdir -p ${OUTDIR}/class_iid/AugMem_ResNet18/
mkdir -p plots

python -u experiment_aug.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT --lr 0.001 --replay_coef 5 --output_dir outputs  --first_times 1 --replay_times 1 --reg_coef 1000 --n_epoch 1  --memory_size 200 --freeze_feature_extract --n_runs 1 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log
python -u plot.py --n_class_per_task 5 --scenario class_iid --output_dir $OUTDIR --result_dir AugMem_ResNet18
mv plots/AugMem_class_iid.png ${OUTDIR}/class_iid/AugMem_ResNet18/AugMem_class_iid.png