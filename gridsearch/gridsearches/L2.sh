python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.001,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.001,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.001,reg_coef-500.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.0001,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.0001,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-0.0001,reg_coef-500.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-1e-05,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-1e-05,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_iid --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_iid/L2/lr-1e-05,reg_coef-500.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.001,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.001,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.001,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.001,reg_coef-500.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.0001,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.0001,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-0.0001,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-0.0001,reg_coef-500.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 1.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-1.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-1e-05,reg_coef-1.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 100.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-100.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-1e-05,reg_coef-100.0/log.log
wait 
python -u experiment.py --scenario class_instance --n_runs 2 --model_type resnet --model_name ResNet18 --agent_type regularization --agent_name L2 --optimizer SGD --batch_size 20 --lr 1e-05 --momentum 0.9 --weight_decay 0.0001 --pretrained --n_epoch 1 --reg_coef 500.0 --memory_size 15 --dataroot /media/data/morgan_data/toybox/images --filelist_root dataloaders --dataset toybox --output_dir toybox_gridsearch_outputs --n_workers 20 --gpuid 1 --validate --custom_folder L2/lr-1e-05,reg_coef-500.0/ | tee ./toybox_gridsearch_outputs/class_instance/L2/lr-1e-05,reg_coef-500.0/log.log
