import logging.config
import os
from typing import List

import PIL
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
#from utils.cifar100 import cifar100

logger = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None):
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)
        if self.dataset == "toybox":
            dataroot = '/media/data/morgan_data/toybox/images'
            img_path = os.path.join(dataroot, img_name)
        if self.dataset == "ilab":
            dataroot = '/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed' 
            img_path = os.path.join(dataroot, img_name)
        if self.dataset == "core50":
            dataroot = '/media/data/Datasets/Core50/core50_128x128'
            img_path = os.path.join(dataroot, img_name)
        if self.dataset == "cifar100":
            #dataroot = '/media/data/morgan_data/toybox/images'
            img_path =  os.path.join("dataset", self.dataset, img_name)

        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        sample["image_name"] = img_name
        return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]


def get_train_datalist(args, cur_iter: int) -> List:
    '''
    if args.mode == "joint":
        datalist = []
        for cur_iter_ in range(args.n_tasks):
            collection_name = get_train_collection_name(
                dataset=args.dataset,
                exp=args.exp_name,
                rnd=args.rnd_seed,
                n_cls=args.n_cls_a_task,
                iter=cur_iter_,
            )
            print("before error?")
            print("collections", args.dataset,collection_name, "json")
            datalist += pd.read_json(
                f"collections/{args.dataset}/{collection_name}.json"
            ).to_dict(orient="records")
            logger.info(f"[Train] Get datalist from {collection_name}.json")
    else:
        collection_name = get_train_collection_name(
            dataset=args.dataset,
            exp=args.exp_name,
            rnd=args.rnd_seed,
            n_cls=args.n_cls_a_task,
            iter=cur_iter,
        )
        print("before error?")
        print("collections", args.dataset,collection_name, "json")
        datalist = pd.read_json(
            f"collections/{args.dataset}/{collection_name}.json"
        ).to_dict(orient="records")
        logger.info(f"[Train] Get datalist from {collection_name}.json")
    print("datalist")
    print(type(datalist))
    print(datalist)
    '''
    if args.dataset == "toybox":
        datalist, _ =  toybox('class_instance',args.run) #cifar100('class_iid',0)
    if args.dataset == "ilab":
        datalist, _ =  ilab('class_instance',args.run)
    if args.dataset == "core50":
        #datalist, _ =  core50('class_iid',args.run)
        datalist, _ =  core50('class_instance',args.run)
        print("class_instance")
    if args.dataset == "cifar100":
        datalist, _ =  cifar100('class_iid',args.run)
    #print("datalist")
    #print(type(datalist))
    #print("cur_iter",cur_iter)
    #print(datalist[cur_iter])
    return datalist[cur_iter]


def get_train_collection_name(dataset, exp, rnd, n_cls, iter):
    collection_name = "{dataset}_train_{exp}_rand{rnd}_cls{n_cls}_task{iter}".format(
        dataset=dataset, exp=exp, rnd=rnd, n_cls=n_cls, iter=iter
    )
    return collection_name


def get_test_datalist(args, exp_name: str, cur_iter: int) -> List:
    '''
    if exp_name is None:
        exp_name = args.exp_name

    if exp_name in ["joint", "blurry10", "blurry30"]:
        # merge over all tasks
        tasks = list(range(args.n_tasks))
    elif exp_name == "disjoint":
        # merge current and all previous tasks
        tasks = list(range(cur_iter + 1))
    else:
        raise NotImplementedError

    datalist = []
    for iter_ in tasks:
        collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
            dataset=args.dataset, rnd=args.rnd_seed, n_cls=args.n_cls_a_task, iter=iter_
        )
        datalist += pd.read_json(
            f"collections/{args.dataset}/{collection_name}.json"
        ).to_dict(orient="records")
        logger.info(f"[Test ] Get datalist from {collection_name}.json")
    '''
    
    if args.dataset == "toybox":
        datalist, test_datalist = toybox('class_iid',0) #here 0 stands for run number which is as of now 0th one #cifar100('class_iid',0)
    if args.dataset == "ilab":
        datalist, test_datalist = ilab('class_iid',0) #here 0 stands for run number which is as of now 0th one
    if args.dataset == "core50":
        datalist, test_datalist = core50('class_iid',0) #here 0 stands for run number which is as of now 0th one
    if args.dataset == "cifar100":
        datalist, test_datalist = cifar100('class_iid',0) #here 0 stands for run number which is as of now 0th one
    
    
    #print("test_datalist")
    #print(type(test_datalist))
    #print("cur_iter",cur_iter)
    #print(test_datalist[cur_iter])
    test_all_till_now = []
    for i in range(cur_iter + 1):
        test_all_till_now = test_all_till_now + test_datalist[i]

    #return test_datalist[cur_iter]
    return test_all_till_now


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "TinyImagenet",
        "toybox",
        "ilab",
        "core50",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "TinyImagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
        "toybox": (0.485, 0.456, 0.406),
        "ilab": (0.485, 0.456, 0.406),
        "core50": (0.5071,0.4866,0.4409),
        
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "TinyImagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
        "toybox": (0.229, 0.224, 0.225),
        "ilab": (0.229, 0.224, 0.225),
        "core50": (0.2673,0.2564,0.2762),
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "TinyImagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
        "toybox": 12,
        "ilab": 14,
        "core50": 10,
    }

    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "TinyImagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
        "toybox": 3,
        "ilab": 3,
        "core50": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 128,
        "CINIC10": 32,
        "TinyImagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
        "toybox": 128,
        "ilab": 128,
        "core50": 128,

    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



######################################################3
def cifar100(paradigm, run):
	batch_num = 20
	rootdir = '/home/rushikesh/code/dataloaders/cifar100_task_filelists/'

	rain_train = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	rain_test = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	train_data = []
	train_labels = []
    #test+labels = [append((path, int(label)))]
	train_groups = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for b in range(batch_num):
		with open( rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
			#print("opened successfully")
			for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					#path = dataroot + path
					train_groups[b].append((path, int(label)))
					train_data.append(path)
					train_labels.append(int(label))
					rain_train[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
	train = {'data': train_data,'fine_labels': train_labels}
	val_groups = train_groups.copy()        
			   
	test_data = []
	test_labels = []
	test_groups = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	idss = [i for i in range(100)]

	groupsid = {'0':0,'1':0,'2':0,'3':0,'4':0,
	'5':1,'6':1,'7':1,'8':1,'9':1,
	'10':2,'11':2,'12':2,'13':2,'14':2,
	'15':3,'16':3,'17':3,'18':3,'19':3,
	'20':4,'21':4,'22':4,'23':4,'24':4,
	'25':5,'26':5,'27':5,'28':5,'29':5,
	'30':6,'31':6,'32':6,'33':6,'34':6,
	'35':7,'36':7,'37':7,'38':7,'39':7,
	'40':8,'41':8,'42':8,'43':8,'44':8,
	'45':9,'46':9,'47':9,'48':9,'49':9,
	'50':10,'51':10,'52':10,'53':10,'54':10,
	'55':11,'56':11,'57':11,'58':11,'59':11,
	'60':12,'61':12,'62':12,'63':12,'64':12,
	'65':13,'66':13,'67':13,'68':13,'69':13,
	'70':14,'71':14,'72':14,'73':14,'74':14,
	'75':15,'76':15,'77':15,'78':15,'79':15,
	'80':16,'81':16,'82':16,'83':16,'84':16,
	'85':17,'86':17,'87':17,'88':17,'89':17,
	'90':18,'91':18,'92':18,'93':18,'94':18,
	'95':19,'96':19,'97':19,'98':19,'99':19	}
	with open( rootdir + paradigm + '/run' + str(run) + '/stream/test_filelist.txt','r') as f:
		for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					gp = groupsid[label]
					#print("label, gp")
					#print(label, gp)
					test_groups[gp].append((path, int(label)))
					test_data.append(path)
					test_labels.append(int(label))
					rain_test[gp].append({'klass': str(label), 'file_name': path, 'label': int(label)})


	#test = {'data': test_data,'fine_labels': test_labels}
	#print("rain_test")
	#print(rain_test)
    #print("rain_test")
	return rain_train,rain_test

def getNextClasses(test_id):

	return train_groups[test_id], val_groups[test_id], test_groups[test_id]   #self.test_grps #



def toybox(paradigm, run):
	batch_num = 6
	rootdir = '/home/rushikesh/code/dataloaders/toybox_task_filelists/'
	rain_train = [[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	rain_test = [[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	train_data = []
	train_labels = []
    #test+labels = [append((path, int(label)))]
	train_groups = [[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for b in range(batch_num):
		with open( rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
			#print("opened successfully")
			for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					
					train_groups[b].append((path, int(label)))
					train_data.append(path)
					train_labels.append(int(label))
					rain_train[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
	train = {'data': train_data,'fine_labels': train_labels}
	val_groups = train_groups.copy()        
			   
	test_data = []
	test_labels = []
	test_groups = [[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	idss = [i for i in range(100)]

	groupsid = {'0':0,'1':0,
					'2':1,'3':1,
					'4':2,'5':2,
					'6':3,'7':3,
					'8':4,'9':4,
					'10':5,'11':5}
	with open( rootdir + paradigm + '/run' + str(run) + '/stream/test_filelist.txt','r') as f:
		for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					gp = groupsid[label]
					#print("label, gp")
					#print(label, gp)
					test_groups[gp].append((path, int(label)))
					test_data.append(path)
					test_labels.append(int(label))
					rain_test[gp].append({'klass': str(label), 'file_name': path, 'label': int(label)})


	#test = {'data': test_data,'fine_labels': test_labels}
	#print("rain_test")
	#print(rain_test)
    #print("rain_test")
	return rain_train,rain_test



def ilab(paradigm, run):
	batch_num = 7
	rootdir = '/home/rushikesh/code/dataloaders/ilab2mlight_task_filelists/'
	rain_train = [[],[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	rain_test = [[],[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	train_data = []
	train_labels = []
    #test+labels = [append((path, int(label)))]
	train_groups = [[],[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for b in range(batch_num):
		with open( rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
			#print("opened successfully")
			for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					
					train_groups[b].append((path, int(label)))
					train_data.append(path)
					train_labels.append(int(label))
					rain_train[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
	train = {'data': train_data,'fine_labels': train_labels}
	val_groups = train_groups.copy()        
			   
	test_data = []
	test_labels = []
	test_groups = [[],[],[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	idss = [i for i in range(100)]

	groupsid = {'0':0,'1':0,
					'2':1,'3':1,
					'4':2,'5':2,
					'6':3,'7':3,
					'8':4,'9':4,
					'10':5,'11':5,
                    '12':6,'13':6}
	with open( rootdir + paradigm + '/run' + str(run) + '/stream/test_filelist.txt','r') as f:
		for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					gp = groupsid[label]
					#print("label, gp")
					#print(label, gp)
					test_groups[gp].append((path, int(label)))
					test_data.append(path)
					test_labels.append(int(label))
					rain_test[gp].append({'klass': str(label), 'file_name': path, 'label': int(label)})


	return rain_train,rain_test



def core50(paradigm, run):
	batch_num = 5
	rootdir = '/home/rushikesh/code/core50_dataloaders/dataloaders/task_filelists/'
	rain_train = [[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	rain_test = [[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	train_data = []
	train_labels = []
	train_groups = [[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for b in range(batch_num):
		with open( rootdir + paradigm + '/run' + str(run) + '/stream/train_task_' + str(b).zfill(2) + '_filelist.txt','r') as f:
			#print("opened successfully")
			for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					
					train_groups[b].append((path, int(label)))
					train_data.append(path)
					train_labels.append(int(label))
					rain_train[b].append({'klass': str(label), 'file_name': path, 'label': int(label)})
	train = {'data': train_data,'fine_labels': train_labels}
	val_groups = train_groups.copy()        
			   
	test_data = []
	test_labels = []
	test_groups = [[],[],[],[],[]] #,[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	

	groupsid = {'0':0,'1':0,
					'2':1,'3':1,
					'4':2,'5':2,
					'6':3,'7':3,
					'8':4,'9':4}
	with open( rootdir + paradigm + '/run' + str(run) + '/stream/test_filelist.txt','r') as f:
		for i, line in enumerate(f):
				if line.strip():
					path, label = line.split()
					gp = groupsid[label]
					#print("label, gp")
					#print(label, gp)
					test_groups[gp].append((path, int(label)))
					test_data.append(path)
					test_labels.append(int(label))
					rain_test[gp].append({'klass': str(label), 'file_name': path, 'label': int(label)})


	return rain_train,rain_test