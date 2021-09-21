MY_PYTHON="python"

seed=0

echo "Hello World"

#$MY_PYTHON main.py  --dataset ilab --epoch 15 --lr 0.0001  --total_cls 14 --numrun 1 --paradigm "class_instance"



echo "Hello World"
$MY_PYTHON main.py  --dataset core50 --epoch 1 --lr 0.0001  --total_cls 10 --numrun 1 --paradigm "class_instance"




#echo "Hello World"
#$MY_PYTHON main.py  --dataset core50 --epoch 15 --lr 0.0001  --total_cls 10 --numrun 10 --paradigm "class_iid"



