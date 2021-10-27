MY_PYTHON="python"

seed=0

#mengmi - class iid
for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON -m stable_sgd.main --dataset 'toybox' --tasks 6 --epochs-per-task 6 --lr 0.15 --gamma 0.85 --batch-size 128 --dropout 0.1 --seed 1234 --paradigm 'class_instance' --run $i
done

