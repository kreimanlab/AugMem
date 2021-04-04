OUTDIR = outputs
NOW = $(date)
mkdir -p ${OUTDIR}/iid/
mkdir -p ${OUTDIR}/class_iid/
mkdir -p ${OUTDIR}/instance/
mkdir -p ${OUTDIR}/class_instance/
python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name L2  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/iid/L2_ResNet18_log.log                 &
python -u experiment.py --scenario class_iid          --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name L2  --gpuid 1 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_iid/L2_ResNet18_log.log           &
python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name L2  --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/instance/L2_ResNet18_log.log            &
python -u experiment.py --scenario class_instance     --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name L2  --gpuid 3 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_instance/L2_ResNet18_log.log      &
python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/iid/EWC_ResNet18_log.log                 &
python -u experiment.py --scenario class_iid          --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC --gpuid 1 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_iid/EWC_ResNet18_log.log           &
python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/instance/EWC_ResNet18_log.log            &
python -u experiment.py --scenario class_instance     --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name EWC --gpuid 3 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/class_instance/EWC_ResNet18_log.log      &