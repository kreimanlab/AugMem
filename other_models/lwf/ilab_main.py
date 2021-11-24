from LwF import LwFmodel
from ResNet import resnet18_cbam
parser=1
numclass= 2 #int(10/parser)
task_size= 2 #int(10/parser)
feature_extractor=resnet18_cbam()
img_size=128
batch_size=128
#task_size=int(10/parser)
memory_size=2000
epochs=10
paradigm = 'class_instance'
dataset = 'ilab'
tasks = 7
results = []




print("ilab 1st run")
learning_rate=0.2

for run in range(10):
    epochs=10
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        results.append(accuracy)
        model.afterTrain(accuracy)
        epochs=1

df = pd.DataFrame(results)
df.to_csv("15Nov_lwf_ilab_avg_results.csv", sep='\t',index=False)


'''
print("ilab 2nd run")

learning_rate=0.02

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)
    epochs=10

    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1


print("ilab 3rd run")

learning_rate=0.002

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1


print("ilab 4th run")

memory_size=2000
epochs=20
learning_rate=0.0002

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1


print("ilab 5th run")

memory_size=2000
epochs=20
learning_rate=2.0

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1
'''