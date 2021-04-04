#!/bin/bash
OUTDIR="outputs_NoReg"
#mkdir -p $OUTDIR/iid/AugMem_ResNet18/
mkdir -p $OUTDIR/class_iid/AugMem_ResNet18/
#mkdir -p $OUTDIR/instance/AugMem_ResNet18/
mkdir -p $OUTDIR/class_instance/AugMem_ResNet18/

#(224*224*3+10)*1200/[ ((64*13*13)+10)*400+1*250*8 ] = 83
#[ ((64*13*13)+10)*400+1*250*8 ]/(224*224*3+10) = 28

#--freeze_feature_extract
#--freeze_batchnorm
#--freeze_memory
#--model_weights ./pretrain/cifar
#--freeze_memory
#--visualize
#python -u experiment.py --scenario iid --reg_coef 1000 --output_dir outputs_rohil --n_epoch 1 --memory_size 200 --freeze_feature_extract --n_runs 10 --model_type resnet --memory_size 200 --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/iid/AugMem_ResNet18/log.log	#&

#python -u experiment.py --scenario instance --reg_coef 1000 --output_dir outputs_rohil --n_epoch 1 --memory_size 200 --model_weights ./pretrain/cifar --freeze_feature_extract --memory_size 200 --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/instance/AugMem_ResNet18/log.log    #&  
    

#python -u experiment_aug.py --scenario class_iid --replay_coef 5 --output_dir outputs_NoReg  --first_times 1 --replay_times 1 --reg_coef 0 --n_epoch 1  --memory_size 200 --freeze_feature_extract --n_runs 5 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log    #&            
#--freeze_feature_extract

python -u experiment_aug.py --freeze_feature_extract --lr 0.00001 --first_times 3 --output_dir outputs_NoReg --replay_coef 5 --replay_times 1 --memory_size 200 --scenario class_instance --reg_coef 0 --n_epoch 1 --n_runs 5 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_instance/AugMem_ResNet18/log.log    #&   
