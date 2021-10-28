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

mkdir -p ${OUTDIR}/class_iid/EWC_ResNet18/

python -u experiment.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT  --lr 0.0001 --reg_coef 1 --n_epoch_first_task 10 --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/EWC_ResNet18/log.log
