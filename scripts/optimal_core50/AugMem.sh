# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
# ./scripts/AugMem_toybox.sh toybox 1

DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/data/Datasets/Core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p $OUTDIR/iid/AugMem_SqueezeNet/
mkdir -p $OUTDIR/class_iid/AugMem_SqueezeNet/
#mkdir -p $OUTDIR/instance/AugMem_SqueezeNet/
mkdir -p $OUTDIR/class_instance/AugMem_SqueezeNet/

#python -u experiment.py --scenario iid --reg_coef 1000 --output_dir outputs_rohil --n_epoch 1 --memory_size 200 --freeze_feature_extract --n_runs 10 --model_type resnet --memory_size 200 --model_name SqueezeNet --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/iid/AugMem_SqueezeNet/log.log	#&

#python -u experiment.py --scenario instance --reg_coef 1000 --output_dir outputs_rohil --n_epoch 1 --memory_size 200 --model_weights ./pretrain/cifar --freeze_feature_extract --memory_size 200 --n_runs 10 --model_type resnet --model_name SqueezeNet --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/instance/AugMem_SqueezeNet/log.log    #&

python -u experiment_aug.py --scenario class_iid --n_runs 10 --n_epoch_first_task 10 --n_epoch 1 --lr 0.001 --memory_Nslots 100  --memory_Nfeat 8 --memory_size 200 --replay_times 1 --replay_coef 5 --reg_coef 1000 --freeze_feature_extract --model_type resnet --model_name SqueezeNet --pretrained --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/AugMem_SqueezeNet/log.log    #&


python -u experiment_aug.py --scenario class_instance --n_runs 10 --n_epoch_first_task 10 --n_epoch 1 --lr 0.00025 --memory_Nslots 100  --memory_Nfeat 8 --memory_size 200 --replay_times 1 --replay_coef 5 --reg_coef 1000 --freeze_feature_extract --model_type resnet --model_name SqueezeNet --pretrained --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/AugMem_SqueezeNet/log.log    #&
