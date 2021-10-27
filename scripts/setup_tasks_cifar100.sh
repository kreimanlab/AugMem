# cifar100 task setup
# n_instance parameter: 1 session per object (training session), 1 object per class (not divided into objects), 10 classes per incremental task, thus 10 instances per task
# task_size_iid parameter: 1 object per class, 1 session per object, 500 images per session. 10 classes per incremental task
# Or, more simply: 10 classes per task, 500 training images per class.
python dataloaders/task_setup.py --dataset cifar100 --scenario class_iid --n_class 5 --task_size_iid 2500 --n_instance 5 --test_sess 1