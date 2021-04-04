OUTDIR=outputs
mkdir -p ${OUTDIR}/iid/
mkdir -p ${OUTDIR}/class_iid/
mkdir -p ${OUTDIR}/instance/
mkdir -p ${OUTDIR}/class_instance/
python -u experiment.py --scenario iid                --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --offline --gpuid 0 --n_epoch 40 --momentum 0.9 --weight_decay 0.0001 --batch_size 256 | tee ${OUTDIR}/iid/NormalNN_ResNet18_offline_log.log                 &
python -u experiment.py --scenario class_iid          --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --offline --gpuid 1 --n_epoch 40 --momentum 0.9 --weight_decay 0.0001 --batch_size 256 | tee ${OUTDIR}/class_iid/NormalNN_ResNet18_offline_log.log           &
python -u experiment.py --scenario instance           --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --offline --gpuid 2 --n_epoch 40 --momentum 0.9 --weight_decay 0.0001 --batch_size 256 | tee ${OUTDIR}/instance/NormalNN_ResNet18_offline_log.log            &
python -u experiment.py --scenario class_instance     --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --offline --gpuid 3 --n_epoch 40 --momentum 0.9 --weight_decay 0.0001 --batch_size 256 | tee ${OUTDIR}/class_instance/NormalNN_ResNet18_offline_log.log      &