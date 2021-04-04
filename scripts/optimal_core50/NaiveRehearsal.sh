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

#mkdir -p ${OUTDIR}/iid/NaiveRehearsal_ResNet18/
mkdir -p ${OUTDIR}/class_iid/NaiveRehearsal_ResNet18/
#mkdir -p ${OUTDIR}/instance/NaiveRehearsal_ResNet18/
mkdir -p ${OUTDIR}/class_instance/NaiveRehearsal_ResNet18/

#python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name NaiveRehearsal  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/iid/NaiveRehearsal_ResNet18/log.log  &
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_iid      --lr 0.0001    --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name NaiveRehearsal  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/NaiveRehearsal_ResNet18/log.log           #&
#python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name NaiveRehearsal  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/instance/NaiveRehearsal_ResNet18/log.log            &
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_instance  --lr 0.0001   --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name NaiveRehearsal  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/NaiveRehearsal_ResNet18/log.log      
