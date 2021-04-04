# ilab2mlight task setup
# n_instance parameter: 6 sessions per object (after taking 2 out for testing), 28 objects per class, 2 classes per incremental task, thus 336 instances (sessions) per task
# task_size_iid parameter: 28 objects per class, 6 sessions per object, 15 images per session, 2 classes per incremental task, thus 5040 examples per task
python dataloaders/task_setup.py --dataset ilab2mlight --scenario iid --task_size_iid 5040 --n_instance 336 --test_sess 4 8            --offline
python dataloaders/task_setup.py --dataset ilab2mlight --scenario class_iid --task_size_iid 5040 --n_instance 336 --test_sess 4 8            --offline
python dataloaders/task_setup.py --dataset ilab2mlight --scenario instance --task_size_iid 5040 --n_instance 336 --test_sess 4 8            --offline
python dataloaders/task_setup.py --dataset ilab2mlight --scenario class_instance --task_size_iid 5040 --n_instance 336 --test_sess 4 8            --offline
python dataloaders/task_setup.py --dataset ilab2mlight --scenario iid --task_size_iid 5040 --n_instance 336 --test_sess 4 8
python dataloaders/task_setup.py --dataset ilab2mlight --scenario class_iid --task_size_iid 5040 --n_instance 336 --test_sess 4 8
python dataloaders/task_setup.py --dataset ilab2mlight --scenario instance --task_size_iid 5040 --n_instance 336 --test_sess 4 8
python dataloaders/task_setup.py --dataset ilab2mlight --scenario class_instance --task_size_iid 5040 --n_instance 336 --test_sess 4 8