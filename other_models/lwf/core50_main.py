from LwF import LwFmodel
from ResNet import resnet18_cbam
print(" Starting experiment on Core50")
parser=1
numclass= 2 #int(10/parser)
task_size= 2 #int(10/parser)
feature_extractor=resnet18_cbam()
img_size=128
batch_size=128
#task_size=int(10/parser)


memory_size=200
epochs=10
learning_rate=0.2
paradigm = 'class_instance'
dataset = 'core50'
tasks = 5

print("core50 1st run")
for run in range(10):
    epochs=10
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)
    
    print("current run is")
    print(run)
    
    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1

'''
print("core50 2nd run")

memory_size=1000
epochs=10
learning_rate=0.02
paradigm = 'class_iid'
dataset = 'core50'
tasks = 5


for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)
    epochs=10

    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1
'''
'''

print("core50 3rd run")
memory_size=2000
epochs=20
learning_rate=0.002
paradigm = 'class_iid'
dataset = 'core50'
tasks = 5

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):
        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1



print("core50 4th run")

memory_size=2000
epochs=20
learning_rate=0.0002
paradigm = 'class_iid'
dataset = 'core50'
tasks = 5

print("core50 1st run")

for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1
'''

'''
print("core50 5th run")

memory_size=2000
epochs=20
learning_rate=2.0
paradigm = 'class_iid'
dataset = 'core50'
tasks = 5


for run in range(10):
    model=LwFmodel(numclass,feature_extractor,batch_size,epochs,learning_rate,task_size,paradigm,dataset)

    for i in range(tasks):

        model.beforeTrain(i+1,run)
        accuracy=model.train(i+1)
        model.afterTrain(accuracy)
        epochs=1
'''