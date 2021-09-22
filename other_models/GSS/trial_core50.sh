results="./output/core50"
CIFAR_10i="--data_path ./data/ --save_path $results --batch_size 10 --log_every 10 --data_file core50.pt   --tasks_to_preserve 5        --cuda yes "

mkdir $results
MY_PYTHON="python"

seed=0




#mengmi - class iid
for i in {0..9}
do
echo "Hello World class_iid"
echo $i
$MY_PYTHON core50_main.py $CIFAR_10i --model GSS_Greedy  --n_epochs 10 --lr 0.001  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed $seed  --subselect 1 --run $i --paradigm "class_iid"

done


for i in {0..9}
do
echo "Hello World class_instance"
echo $i
$MY_PYTHON core50_main.py $CIFAR_10i --model GSS_Greedy  --n_epochs 10 --lr 0.001  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed $seed  --subselect 1 --run $i --paradigm "class_instance"

done


