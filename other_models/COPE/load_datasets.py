import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import pdb
import numpy as np
import pandas as pd

import torch
from metrics.metrics import confusion_matrix

from torchvision import transforms
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torchvision.datasets.folder import pil_loader

def load_datasets(args):

    if args.dataset == 'core50':
        args.n_tasks = 5    
        rootdir = '/home/rushikesh/code/core50_dataloaders/dataloaders/task_filelists/'
        dataroot = '/media/data/Datasets/Core50/core50_128x128'
        
        input_transform= Compose([
            transforms.Resize(128),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
       
        #if os.path.exists('core50.pt'):
            #x_tr, y_tr, x_te, y_te = torch.load('core50.pt')  
        if os.path.exists('core50' + str(args.run) +'_' + args.paradigm + '.pt'):
            x_tr, y_tr, x_te, y_te = torch.load('core50' + str(args.run) +'_' + args.paradigm +'.pt') 



        else:
            print('started loading train data')
            
            #loading training data
            x_tr = []
            y_tr = []
            for t in range(args.n_tasks):
                with open(rootdir + args.paradigm + '/run' + str(args.run) + '/stream/train_task_' + str(t).zfill(2) + '_filelist.txt','r') as f:
                    for i, line in enumerate(f):
                        #print('train: ' + str(i))
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
            #torch.save((x_tr, y_tr, x_te, y_te), 'core50.pt')
            torch.save((x_tr, y_tr, x_te, y_te),'core50' + str(args.run) +'_' + args.paradigm + '.pt')
            print('finished saving data')
        
        #convert list to tensor
        x_tr = torch.stack(x_tr).squeeze()
        #print(x_tr.shape)
        #print(y_tr)
        y_tr = torch.LongTensor(y_tr)
        
        x_te = torch.stack(x_te)
        y_te = torch.LongTensor(y_te)
        
        cpt = int(10 / args.n_tasks)
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
        return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


    elif args.dataset == 'toybox':

        args.n_tasks = 6
        #rootdir = '/home/rushikesh/code/dataloaders/toybox_task_filelists/'
        rootdir = '/home/rushikesh/P1_Oct/toybox_task_filelists/'
        dataroot = '/media/data/morgan_data/toybox/images'

        input_transform= Compose([
            transforms.Resize(32),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
       
        if os.path.exists('toybox' + str(args.run) +'_' + args.paradigm + '.pt'):
            x_tr, y_tr, x_te, y_te = torch.load('toybox' + str(args.run) +'_' + args.paradigm + '.pt')
        else:
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
        
        #convert list to tensor
        x_tr = torch.stack(x_tr).squeeze()
        y_tr = torch.LongTensor(y_tr)
        
        x_te = torch.stack(x_te)
        y_te = torch.LongTensor(y_te)
        
        cpt = int(12 / args.n_tasks)
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
        return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


    elif args.dataset == 'ilab':

        args.n_tasks = 7
        #rootdir = '/home/rushikesh/code/dataloaders/ilab2mlight_task_filelists/'
        rootdir = '/home/rushikesh/P1_Oct/ilab2mlight_task_filelists/'
        dataroot = '/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed'

        input_transform= Compose([
            transforms.Resize(32),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
       

        if os.path.exists('ilab' + str(args.run) +'_' + args.paradigm + '.pt'):
            x_tr, y_tr, x_te, y_te = torch.load('ilab' + str(args.run) +'_' + args.paradigm +'.pt') 
        else:
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
            torch.save((x_tr, y_tr, x_te, y_te),'ilab' + str(args.run) +'_' + args.paradigm + '.pt')
            print('finished saving data')
        
        #convert list to tensor
        x_tr = torch.stack(x_tr).squeeze()
        y_tr = torch.LongTensor(y_tr)
        
        x_te = torch.stack(x_te)
        y_te = torch.LongTensor(y_te)
        
        cpt = int(14/ args.n_tasks)
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
        return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)

    elif args.dataset == 'mini_imagenet':
        print("here in mini_imagenet in load_dataset.py")

 
        args.n_tasks = 20   
        #rootdir = '/home/rushikesh/code/dataloaders/toybox_task_filelists/'
        #dataroot = '/media/data/morgan_data/toybox/images'

        input_transform= Compose([
            transforms.Resize(32),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        weights_path = '/media/data/morgan_data/mini_imagenet/sampled_miniimagenet_train.pt'

        if os.path.exists(weights_path): #'toybox' + str(args.run) +'_' + args.para>
            x_tr, y_tr, x_te, y_te = torch.load(weights_path) #'toybox' + str(args.>
            
            x_tr, y_tr, x_te, y_te =  model.load_state_dict(weights)
        
        else:
            print(".pt file not found")
            
            
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
            torch.save((x_tr, y_tr, x_te, y_te), 'mini_imagenet' + str(args.run) +'_' + args.paradigm + '.pt')
            print('finished saving data')
            
        
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
        return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


    elif args.dataset == 'cifar100':

        args.n_tasks = 20
        rootdir = '/home/rushikesh/code/dataloaders/cifar100_task_filelists/'
        dataroot = '/home/rushikesh/P1_Oct/cifar100/cifar100png'

        input_transform= Compose([
            transforms.Resize(32),
            ToTensor(),
            Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
       

        if os.path.exists('cfiar100' + str(args.run) +'_' + args.paradigm + '.pt'):
            x_tr, y_tr, x_te, y_te = torch.load('ilab' + str(args.run) +'_' + args.paradigm +'.pt') 
        else:
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
            torch.save((x_tr, y_tr, x_te, y_te),'cifar100' + str(args.run) +'_' + args.paradigm + '.pt')
            print('finished saving data')
        
        #convert list to tensor
        x_tr = torch.stack(x_tr).squeeze()
        y_tr = torch.LongTensor(y_tr)
        
        x_te = torch.stack(x_te)
        y_te = torch.LongTensor(y_te)
        
        cpt = int(100/ args.n_tasks)
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
        return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)