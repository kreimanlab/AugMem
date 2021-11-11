# Param #1: dataset name, e.g. core50, toybox, ilab2mlight, cifar100. Default is core50
# Param #2: GPU ID. Default is 0
# Usage example: ./scripts/AugMem.sh cifar100 0 0.001 1 100 8 /media/data/morgan_data/cifar100
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"
lr=${3:-0.001}
mem_sparse=${4:-1}
memory_Nslots=${5:-100}
memory_Nfeat=${6:-8}


if [ "$DATASET" = "core50" ]; then
    DDATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DDATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DDATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
    #DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/ilab/iLab-2M-Light/"
elif [ "$DATASET" = "cifar100" ]; then
    DDATAROOT="./data/cifar100"
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${7:-${DDATAROOT}}

OUTDIR=augmem_gridsearch_cifar100
custom_folder="AugMem_lr_${lr}_memsparse_${mem_sparse}_memNslots_${memory_Nslots}"
mkdir -p ${OUTDIR}/class_iid/${custom_folder}
mkdir -p plots

python -u experiment_aug.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT --n_runs 2 --n_epoch 1 --n_epoch_first_task 1 --lr ${lr} --mem_sparse ${mem_sparse} --memory_Nslots ${memory_Nslots} --memory_Nfeat ${memory_Nfeat} --replay_times 1 --replay_coef 5 --reg_coef 1000 --memory_size 200 --freeze_feature_extract --model_type resnet --model_name ResNet18 --pretrained --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 --output_dir ${OUTDIR} --custom_folder ${custom_folder} | tee ${OUTDIR}/class_iid/${custom_folder}/log.log
python -u plot.py --n_class_per_task 5 --scenario class_iid --output_dir $OUTDIR --result_dir ${custom_folder}
mv plots/AugMem_class_iid.png ${OUTDIR}/class_iid/${custom_folder}/AugMem_class_iid.png