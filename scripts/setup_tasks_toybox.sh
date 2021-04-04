# Toybox task setup
# n_instance parameter: 7 sessions per object (after taking 3 out for testing), 29 objects per class, 2 classes per incremental task, thus 406 instances (sessions) per task
# task_size_iid parameter: 29 objects per class, 7 sessions per object, 15 images per session, 2 classes per incremental task, thus 6090 examples per task
python dataloaders/task_setup.py --dataset toybox --scenario iid --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario class_iid --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario instance --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario class_instance --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario iid --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9
python dataloaders/task_setup.py --dataset toybox --scenario class_iid --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9
python dataloaders/task_setup.py --dataset toybox --scenario instance --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9
python dataloaders/task_setup.py --dataset toybox --scenario class_instance --task_size_iid 6090 --n_instance 406 --test_sess 3 6 9
