
results="./output/toybox_upload"
CIFAR_10i="--data_path ./data/ --save_path $results --batch_size 100 --log_every 10 --data_file toybox.pt   --tasks_to_preserve 6        --cuda yes "

mkdir $results
MY_PYTHON="python"

seed=0
#trial for upload
$MY_PYTHON main.py $CIFAR_10i --model GSS_Greedy  --n_epochs 1 --lr 0.0001  --n_iter 1 --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 100  --change_th 0. --seed $seed  --subselect 1 --run 0 --paradigm "class_iid"

'''
#mengmi - class iid
for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON toybox_main.py $CIFAR_10i --model GSS_Greedy  --n_epochs 15 --lr 0.0001  --n_iter 1 --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 100  --change_th 0. --seed $seed  --subselect 1 --run $i --paradigm "class_iid"

done


for i in {0..9}
do
echo "Hello World"
echo $i
$MY_PYTHON toybox_main.py $CIFAR_10i --model GSS_Greedy  --n_epochs 15 --lr 0.0001  --n_iter 10 --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 100  --change_th 0. --seed $seed  --subselect 1 --run $i --paradigm "class_instance"

done
'''



