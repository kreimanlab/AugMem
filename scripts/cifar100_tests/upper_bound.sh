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
    DATAROOT="./data/cifar100"
else
    echo "Invalid dataset name!"
    exit
fi

mkdir -p ${OUTDIR}/class_iid/NormalNN_ResNet18_offline/
mkdir -p plots

python -u experiment.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT  --lr 0.0001 --n_epoch_first_task 10  --n_epoch 10 --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type default --agent_name NormalNN  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 100 --n_workers 8 --output_dir $OUTDIR --keep_best_net_all_tasks --offline | tee ${OUTDIR}/class_iid/NormalNN_ResNet18_offline/log.log
python -u plot.py --n_class_per_task 5 --scenario class_iid --output_dir $OUTDIR --result_dir NormalNN_ResNet18_offline
mv plots/NormalNN_class_iid.png ${OUTDIR}/class_iid/NormalNN_ResNet18_offline/NormalNN_offline_class_iid.png