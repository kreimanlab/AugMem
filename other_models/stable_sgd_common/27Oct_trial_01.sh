MY_PYTHON="python"

seed=0


echo "run1"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run2"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.0015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run3"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.00015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run1"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.75 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run2"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.0015 --gamma 0.75 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run3"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.00015 --gamma 0.75 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0


echo "run1"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.65 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run2"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.0015 --gamma 0.65 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

echo "run3"
$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.00015 --gamma 0.65 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0


#mengmi - class iid
# for i in {0..9}
# do
# echo "Hello World"
# echo $i
#$MY_PYTHON main.py --dataset 'cifar100' --tasks 20 --epochs-per-task 1 --lr 0.15 --gamma 0.85 --batch-size 256 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run $i
# done

