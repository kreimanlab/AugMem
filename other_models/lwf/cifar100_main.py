from LwF import LwFmodel
from ResNet import resnet18_cbam
import pandas as pd
print(" Starting experiment on cifar100")
parser=1
numclass= 5 #int(10/parser)
task_size= 5 #int(10/parser)
feature_extractor=resnet18_cbam()
img_size=128
batch_size=128
#task_size=int(10/parser)

memory_size=200
epochs=10
learning_rate=0.2
paradigm = 'class_iid'
dataset = 'cifar100'
tasks = 20

results = []
print("cifar100 1st run")
for run in range(10):
    epochs=20
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)
    
    print("current run is")
    print(run)
    
    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        results.append(accuracy)
        model.afterTrain(accuracy)
        epochs=1

df = pd.DataFrame(results)
df.to_csv("lwf_cifar100_avg_results.csv", sep='\t',index=False)
