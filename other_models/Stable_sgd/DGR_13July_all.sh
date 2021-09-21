MY_PYTHON="python"

seed=0

#example run
for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON -m main  --replay-mode=generative-replay --generator-iterations 100 --solver-iterations 100 --eval-log-interval 1 --loss-log-interval 1 --sample-log-interval 1 --train --experiment core50 --run $i --paradigm 'class_iid' --num_tasks 5

done

for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON -m main  --replay-mode=generative-replay --generator-iterations 100 --solver-iterations 100 --eval-log-interval 1 --loss-log-interval 1 --sample-log-interval 1 --train --experiment core50 --run $i --paradigm 'class_instance' --num_tasks 5

done




