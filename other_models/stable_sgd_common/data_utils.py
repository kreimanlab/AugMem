import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from batch_data import BatchData
import torchvision.transforms.functional as TorchVisionFunc
from core50 import core50
from toybox import toybox
from ilab import ilab
from cifar100 import cifar100



def get_permuted_mnist(task_id, batch_size):
	"""
	Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
	This function will be called several times for each task.
	
	:param task_id: id of the task [starts from 1]
	:param batch_size:
	:return: a tuple: (train loader, test loader)
	"""
	
	# convention, the first task will be the original MNIST images, and hence no permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
				])
	mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks, batch_size):
	"""
	Returns the datasets for sequential tasks of permuted MNIST
	
	:param num_tasks: number of tasks.
	:param batch_size: batch-size for loaders.
	:return: a dictionary where each key is a dictionary itself with train, and test loaders.
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_permuted_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


class RotationTransform:
	"""
	Rotation transforms for the images in `Rotation MNIST` dataset.
	"""
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size):
	"""
	Returns the dataset for a single task of Rotation MNIST dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	per_task_rotation = 10
	rotation_degree = (task_id - 1)*per_task_rotation
	rotation_degree -= (np.random.random()*per_task_rotation)

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
		])

	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks, batch_size):
	"""
	Returns data loaders for all tasks of rotation MNIST dataset.
	:param num_tasks: number of tasks in the benchmark.
	:param batch_size:
	:return:
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_rotated_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


def get_split_cifar100(task_id, batch_size, cifar_train, cifar_test):
	"""
	Returns a single task of split CIFAR-100 dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	

	start_class = (task_id-1)*5
	end_class = task_id * 5

	targets_train = torch.tensor(cifar_train.targets)
	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
	
	targets_test = torch.tensor(cifar_test.targets)
	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

	train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0]), batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=batch_size)

	return train_loader, test_loader


def get_split_cifar100_tasks(num_tasks, batch_size,run,paradigm,dataset):
	"""
	Returns data loaders for all tasks of split CIFAR-100
	:param num_tasks:
	:param batch_size:
	:return:
	
	datasets = {}
	
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
	
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets
	"""
	"""
	datasets = {}
	paradigm = 'class_iid'
	run = 0
	dataset = core50( paradigm, run)
	for task_id in range(0, num_tasks):
		train_loader, val, test_loader = dataset.getNextClasses(task_id) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets
	"""
	datasets = {}
	#paradigm = 'class_iid'
	#run = 0
	#dataset = load_datasets( paradigm, run)
	if dataset == 'core50':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_core50(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			datasets[task_id] = {'train': train_loader, 'test': test_loader}
	
	if dataset == 'toybox':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_toybox(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			datasets[task_id] = {'train': train_loader, 'test': test_loader}
	
	if dataset == 'ilab':
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_ilab(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			datasets[task_id] = {'train': train_loader, 'test': test_loader}
	
	if dataset == 'cifar100':
		print("in data_utils in cifar100")
		for task_id in range(0, num_tasks):
			train_loader, test_loader = dataset_cifar100(task_id,batch_size,run,paradigm,dataset) #get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
			datasets[task_id] = {'train': train_loader, 'test': test_loader}
	

	


	return datasets


def dataset_ilab(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[],[],[]]
			test_ys = [[],[],[],[],[],[],[]]
			#train_xs = [[],[],[],[],[],[]]
			#train_ys = [[],[],[],[],[],[]]
			input_transform= Compose([
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []

			dataset = ilab( paradigm, run)
			print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y, dataset_name,input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y, dataset,input_transform_eval),batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data


def dataset_core50(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[]]
			test_ys = [[],[],[],[],[]]
			train_xs = [[],[],[],[],[]]
			train_ys = [[],[],[],[],[]]
			input_transform= Compose([
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []
			#for inc_i in range(task_id+1):
				#paradigm = 'class_iid'
				#run = 0
			dataset = core50( paradigm, run)
			print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)
				

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y,dataset_name, input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y, input_transform_eval),
						#batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data

def dataset_toybox(task_id, batch_size,run,paradigm,dataset_name):
			test_xs = [[],[],[],[],[],[]]
			test_ys = [[],[],[],[],[],[]]
			#train_xs = [[],[],[],[],[],[]]
			#train_ys = [[],[],[],[],[],[]]
			input_transform= Compose([
									transforms.Resize(32),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(32,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(32),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []

			dataset = toybox( paradigm, run)
			print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)
				

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y,dataset_name, input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y,dataset, input_transform_eval),
						#batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data



def dataset_cifar100(task_id, batch_size,run,paradigm,dataset_name):
			input_transform= Compose([
									transforms.Resize(128),
									transforms.RandomHorizontalFlip(),
									transforms.RandomCrop(128,padding=4),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			input_transform_eval= Compose([
									transforms.Resize(128),
									ToTensor(),
									Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
			test_accs = []
			#print("fine here in dataset_cifar100")
			dataset = cifar100( paradigm, run)
			print(f"Incremental num : {task_id}")
			train, val, test = dataset.getNextClasses(task_id)
			print(len(train), len(val), len(test))
			train_x, train_y = zip(*train)
			val_x, val_y = zip(*val)
			test_x, test_y = zip(*test)
				

			#if inc_i > 0 :
			#    epoches = 1 #stream learning; see data only once

			train_data = DataLoader(BatchData(train_x, train_y,dataset_name, input_transform),
						batch_size=batch_size, shuffle=True, drop_last=True)
			#val_data = DataLoader(BatchData(val_x, val_y,dataset, input_transform_eval),
						#batch_size=batch_size, shuffle=False)            
			test_data = DataLoader(BatchData(test_x, test_y,dataset_name, input_transform_eval),
						batch_size=batch_size, shuffle=False)
			

			return train_data, test_data
'''
def dataset_mini_imagenet_1():

        args.n_tasks = 20
        #rootdir = '/home/rushikesh/code/dataloaders/toybox_task_filelists/'
        #dataroot = '/media/data/morgan_data/toybox/images'

        input_transform= Compose([
            transforms.Resize(32),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
       
        if os.path.exists('/media/data/morgan_data/mini_imagenet/sampled_miniimagenet_train.pt'): #'toybox' + str(args.run) +'_' + args.paradigm + '.pt'):
            x_tr, y_tr, x_te, y_te = torch.load('/media/data/morgan_data/mini_imagenet/sampled_miniimagenet_train.pt') #'toybox' + str(args.run) +'_' + args.paradigm + '.pt')
        else:
			print(".pt file not found")
'''

'''
			print('started loading train data')
            #loading training data
            x_tr = []
            y_tr = []
            for t in range(args.n_tasks):
                with open(rootdir + args.paradigm + '/run' + str(args.run) + '/stream/train_task_' + str(t).zfill(2) + '_filelist.txt','r') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            fpath, label = line.split()
                            
                            image = pil_loader(os.path.join(dataroot, fpath))
                            image = input_transform(image)
                            image = image.view(1, -1)
                            x_tr.append(image)
                            y_tr.append(int(label))
            
            print('started loading test data')
            #loading test data
            x_te = []
            y_te = []
            with open(rootdir + args.paradigm + '/run' + str(args.run) + '/stream/test_filelist.txt','r') as f:
                for i, line in enumerate(f):
                    #print('test: ' + str(i))
                    if line.strip():
                        fpath, label = line.split()
                        
                        image = pil_loader(os.path.join(dataroot, fpath))
                        image = input_transform(image)
                        image = image.view(1, -1)
                        x_te.append(image)
                        y_te.append(int(label))
            
            print('finished loading test data')
            torch.save((x_tr, y_tr, x_te, y_te), 'toybox' + str(args.run) +'_' + args.paradigm + '.pt')
            print('finished saving data')
'''
'''        
        #convert list to tensor
        x_tr = torch.stack(x_tr).squeeze()
        y_tr = torch.LongTensor(y_tr)
        
        x_te = torch.stack(x_te)
        y_te = torch.LongTensor(y_te)
        
        cpt = int(100 / args.n_tasks)
        d_tr = []
        d_te = []
        
        for t in range(args.n_tasks):
            c1 = t * cpt
            c2 = (t + 1) * cpt
            i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
            i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
            d_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
            d_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])
        
        #print("path",args.data_path + '/' + args.data_file)
        #d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
        n_inputs = d_tr[0][1].size(1)
        n_outputs = 0
        for i in range(len(d_tr)):
            n_outputs = max(n_outputs, d_tr[i][2].max().item())
            n_outputs = max(n_outputs, d_te[i][2].max().item())
        return d_tr, d_te    
		#, n_inputs, n_outputs + 1, len(d_tr)

'''
