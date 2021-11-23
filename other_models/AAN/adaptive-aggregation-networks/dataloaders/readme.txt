python dataloaders/task_setup.py --dataset toybox --scenario class_iid
python dataloaders/task_setup.py --dataset toybox --scenario class_instance
python dataloaders/task_setup.py --dataset toybox --scenario class_iid --offline
python dataloaders/task_setup.py --dataset toybox --scenario class_instance --offline
./scripts/combined_gridsearch_toybox.sh
