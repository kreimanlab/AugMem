MY_PYTHON="python"

seed=0





echo "Hello World classiid"
echo "--dataset cifar100 --epoch 15 --lr 0.001  --total_cls 100 --numrun 10 --paradigm class_iid  "
$MY_PYTHON main.py  --dataset cifar100 --epoch 15 --lr 0.001  --total_cls 100 --numrun 10 --paradigm "class_iid"

done
