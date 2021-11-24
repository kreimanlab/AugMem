MY_PYTHON="python"

seed=0
echo "dropout 0.1t_"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "dropout 0.01"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.01 --seed 1234 --paradigm 'class_iid' --run 0

echo "dropout 0.2"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.2 --seed 1234 --paradigm 'class_iid' --run 0


echo "dropout 0.3"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.3 --seed 1234 --paradigm 'class_iid' --run 0

echo "run1"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.05 --seed 1234 --paradigm 'class_iid' --run 0

echo "dropout 0.0"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.0 --seed 1234 --paradigm 'class_iid' --run 0

