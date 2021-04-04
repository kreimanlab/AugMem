DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_gridsearch_outputs"
mkdir $OUTDIR
GPU1="${2:-0}"
GPU2="${3:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
else
    echo "Invalid dataset name!"
    exit
fi

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC --momentum 0.9 --gpuid $GPU1 --weight_decay 0.0001  --reg_coef 1 100 500  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name EWC

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name L2 --momentum 0.9 --gpuid $GPU1 --weight_decay 0.0001  --reg_coef 1 100 500  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name L2

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS --momentum 0.9 --gpuid $GPU1 --weight_decay 0.0001  --reg_coef 1 100 500  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name MAS

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name SI --momentum 0.9 --gpuid $GPU1 --weight_decay 0.0001  --reg_coef 1 100 500  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name SI

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name iCARL --momentum 0.9 --gpuid $GPU1 --weight_decay 0.0001  --reg_coef 1  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name iCARL

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name GEM --momentum 0.9 --gpuid $GPU2 --weight_decay 0.0001  --reg_coef 1  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name GEM

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type exp_replay --agent_name NaiveRehearsal --momentum 0.9 --gpuid $GPU2 --weight_decay 0.0001  --reg_coef 1  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name NaiveRehearsal

python setup_grid.py --scenario class_iid class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --pretrained --agent_type default --agent_name NormalNN --momentum 0.9 --gpuid $GPU2 --weight_decay 0.0001  --reg_coef 1  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name NormalNNlower

python setup_grid.py --offline --scenario class_iid class_instance --n_runs 2 --model_type resnet --n_epoch 10 --model_name ResNet18 --pretrained --agent_type default --agent_name NormalNN --momentum 0.9 --gpuid $GPU2 --weight_decay 0.0001  --reg_coef 1  --batch_size 20 --lr 0.001 0.0001 0.00001 --output_dir $OUTDIR --dataroot $DATAROOT --dataset $DATASET --n_workers 20 --memory_size 15 --grid_name NormalNNupper

