from LwF import LwFmodel
from ResNet import resnet18_cbam
parser=1
numclass= 2 #int(10/parser)
task_size= 2 #int(10/parser)
feature_extractor=resnet18_cbam()
img_size=128
batch_size=128
#task_size=int(10/parser)


results = []

memory_size=200
epochs= 10 # trial for upload otherwise its 10
learning_rate=0.02
paradigm = 'class_instance'
dataset = 'toybox'
tasks = 6


for run in range(10): # trial for upload otherwise its 10
    epochs= 10 # trial for upload otherwise its 10
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        results.append(accuracy)
        model.afterTrain(accuracy)
        epochs=1


df = pd.DataFrame(results)
df.to_csv("15Nov_lwf_toybox_avg_results.csv", sep='\t',index=False)


