# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p ${OUTDIR}/iid/MAS_ResNet18/
mkdir -p ${OUTDIR}/class_iid/MAS_ResNet18/
#mkdir -p ${OUTDIR}/instance/MAS_ResNet18/
mkdir -p ${OUTDIR}/class_instance/MAS_ResNet18/

python -u experiment.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT   --output_dir $OUTDIR        --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS  --gpuid $GPU --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/MAS_ResNet18/log.log           

python -u experiment.py --scenario class_instance --dataset $DATASET --dataroot $DATAROOT   --output_dir $OUTDIR   --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS  --gpuid $GPU --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/MAS_ResNet18/log.log
