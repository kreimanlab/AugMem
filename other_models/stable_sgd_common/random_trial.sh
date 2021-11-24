MY_PYTHON="python"

seed=0
echo "toybox 0.015"
$MY_PYTHON main.py --dataset 'toybox' --tasks 6 --epochs-per-task 15 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

# echo "0.00015"
# $MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 35 --lr 0.00015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0
