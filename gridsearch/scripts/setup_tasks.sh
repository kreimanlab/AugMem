# Core50 (default parameters are built in)
python dataloaders/task_setup.py --scenario iid               --offline
python dataloaders/task_setup.py --scenario class_iid         --offline
python dataloaders/task_setup.py --scenario instance          --offline
python dataloaders/task_setup.py --scenario class_instance    --offline
python dataloaders/task_setup.py --scenario iid               
python dataloaders/task_setup.py --scenario class_iid         
python dataloaders/task_setup.py --scenario instance
python dataloaders/task_setup.py --scenario class_instance

# Toybox
# n_instance parameter: 7 sessions per object (after taking 3 out for testing), 30 objects per class, 2 classes per incremental task, thus 420 instances (videos?) per task
# task_size_iid parameter: 30 objects per class, 7 sessions per object, 15 images per session, 2 classes per incremental task, thus 6300 examples per task
python dataloaders/task_setup.py --dataset toybox --scenario iid --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario class_iid --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario instance --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario class_instance --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9            --offline
python dataloaders/task_setup.py --dataset toybox --scenario iid --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9
python dataloaders/task_setup.py --dataset toybox --scenario class_iid --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9 
python dataloaders/task_setup.py --dataset toybox --scenario instance --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9
python dataloaders/task_setup.py --dataset toybox --scenario class_instance --task_size_iid 6300 --n_instance 420 --test_sess 3 6 9
