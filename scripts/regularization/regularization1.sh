OUTDIR=outputs
mkdir -p ${OUTDIR}/iid/
mkdir -p ${OUTDIR}/class_iid/
mkdir -p ${OUTDIR}/instance/
mkdir -p ${OUTDIR}/class_instance/
python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name SI  --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/iid/SI_ResNet18_log.log                 &
python -u experiment.py --scenario class_iid          --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name SI  --gpuid 1 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/class_iid/SI_ResNet18_log.log           &
python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name SI  --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/instance/SI_ResNet18_log.log            &
python -u experiment.py --scenario class_instance     --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name SI  --gpuid 3 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/class_instance/SI_ResNet18_log.log      &
python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS --gpuid 3 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/iid/MAS_ResNet18_log.log                 &
python -u experiment.py --scenario class_iid          --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS --gpuid 1 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/class_iid/MAS_ResNet18_log.log           &
python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS --gpuid 2 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/instance/MAS_ResNet18_log.log            &
python -u experiment.py --scenario class_instance     --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --agent_type regularization --agent_name MAS --gpuid 3 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 | tee ${OUTDIR}/class_instance/MAS_ResNet18_log.log      &