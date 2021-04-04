# Core50 task setup (appropriate parameters are built in as defaults to task_setup.py)
python dataloaders/task_setup.py --scenario iid               --offline
python dataloaders/task_setup.py --scenario class_iid         --offline
python dataloaders/task_setup.py --scenario instance          --offline
python dataloaders/task_setup.py --scenario class_instance    --offline
python dataloaders/task_setup.py --scenario iid
python dataloaders/task_setup.py --scenario class_iid
python dataloaders/task_setup.py --scenario instance
python dataloaders/task_setup.py --scenario class_instance