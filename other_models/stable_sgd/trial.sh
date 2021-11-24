MY_PYTHON="python"

seed=0

# for i in {0..9}
# do
# echo "Hello World"
# echo $i
# $MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 20 --lr 0.002 --gamma 0.95 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run $i
# #$MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

# done
# echo "new set of runs"

# for i in {0..9}
# do
# echo "Hello World"
# echo $i
# $MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 20 --lr 0.05 --gamma 0.85 --batch-size 128 --dropout 0.3 --seed 1234 --paradigm 'class_iid' --run $i
# #$MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

# done


#echo "--tasks 20 --epochs-per-task 30 --lr 0.2 --gamma 0.99 --batch-size 128 --dropout 0.01 --seed 1234"
echo "--tasks 20 --epochs-per-task 25 --lr 0.001 --gamma 0.99 --batch-size 64 --dropout 0.05"
for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 25 --lr 0.15 --gamma 0.99 --batch-size 64 --dropout 0.05 --seed 1234 --paradigm 'class_iid' --run $i
#$MY_PYTHON -m stable_sgd.main --dataset 'cifar100' --tasks 20 --epochs-per-task 15 --lr 0.015 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_iid' --run 0

done
# for i in {0..9}
# do
# echo "Hello World"
# echo $i
# $MY_PYTHON -m stable_sgd.main --dataset 'core50' --tasks 5 --epochs-per-task 15 --lr 0.2 --gamma 0.95 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_instance' --run $i

# done

