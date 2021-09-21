import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR

import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pickle
from dataset import BatchData
from model import PreResNet, BiasLayer
#from cifar import Cifar100
from core50 import Core50
from toybox import Toybox
from ilab import Ilab
from exemplar import Exemplar
from copy import deepcopy


class Trainer:
    def __init__(self, total_cls, paradigm, run,dataset):
        self.total_cls = total_cls
        self.seen_cls = 0
        #self.dataset = Cifar100()
        if dataset == "core50":
            self.dataset = Core50(paradigm, run)
            self.test_size = 450
        if dataset == 'toybox':
            self.dataset = Toybox(paradigm, run)
            self.test_size = 2610
        if dataset == "ilab":
            print("in ilab data")
            self.dataset = Ilab(paradigm, run)
            self.test_size = 1680

        self.model = PreResNet(32,total_cls).cuda()
        print(self.model)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.bias_layer5 = BiasLayer().cuda()
        if self.total_cls == 10:
            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5]

        if self.total_cls == 12:
            self.bias_layer6 = BiasLayer().cuda()
            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,self.bias_layer6]

        if self.total_cls == 14:
            print("for ilab data")
            self.bias_layer6 = BiasLayer().cuda()
            self.bias_layer7 = BiasLayer().cuda()
            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,self.bias_layer6, self.bias_layer7]
            
        self.input_transform= Compose([
                                transforms.Resize(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                transforms.Resize(32),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)



    def test(self, testdata):
        print("Testing..")
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        task1 = 0
        task2 = 0
        task3 = 0
        task4 = 0
        task5 = 0
        task6 = 0
        task7 = 0

        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0



        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
            for a,b in zip(label,pred):
                if a == b:
                    if a == 0 or a == 1:
                        task1 += 1
                    elif a == 2 or a == 3:
                        task2 += 1
                    elif a == 4 or a == 5:
                        task3 += 1
                    elif a == 6 or a == 7:
                        task4 += 1
                    elif a == 8 or a == 9:
                        task5 += 1
                    elif a == 10 or a == 11:
                        task6 += 1
                    elif a == 12 or a == 13:
                        task7 += 1

                if a == 0 or a == 1:
                    sum1 += 1
                elif a == 2 or a == 3:
                    sum2 += 1
                elif a == 4 or a == 5:
                    sum3 += 1
                elif a == 6 or a == 7:
                    sum4 += 1
                elif a == 8 or a == 9:
                    sum5 += 1
                elif a == 10 or a == 11:
                    sum6 += 1
                elif a == 12 or a == 13:
                    sum7 += 1

        # print("pred and label")
        # print(pred,label)
        

        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        #print("task wise")
        #print(task1, task2, task3, task4, task5,task6, task7)
        #print("task wise sum")
        #print(sum1, sum2, sum3, sum4,sum5, sum6, sum7) 
        test_size = self.test_size
        #print("test wise accuracies")
        #print(task1 / sum1, task2 / sum2, task3 /sum3, task4 / sum4, task5 / sum5,task6 / sum6, task7 / sum7)
        
        if(sum7 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum6 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum5 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum4 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum3 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum2 == test_size):
            print("task 1 acc:", task1/sum1)
        elif(sum1 == test_size):
            print("task 1 acc:", task1/sum1)
        
        
        
       
        
        # print("correct")
        # print(correct)
        # print("wrong + correct")
        # print(wrong + correct)
        self.model.train()
        print("---------------------------------------------")
        return acc


    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        return



    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)

        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            print(len(train), len(val), len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)

            if inc_i > 0 :
                epoches = 1 #stream learning; see data only once

            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            val_data = DataLoader(BatchData(val_x, val_y, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)            
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)


            # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
            bias_optimizer = optim.Adam(self.bias_layers[inc_i].parameters(), lr=0.001)
            # bias_scheduler = StepLR(bias_optimizer, step_size=70, gamma=0.1)
            exemplar.update(total_cls//dataset.batch_num, (train_x, train_y), (val_x, val_y))

            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform),
                        batch_size=1, shuffle=True, drop_last=False)
            test_acc = []


            for epoch in range(epoches):
                print("---"*50)
                print("Epoch", epoch)
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                for _ in range(len(self.bias_layers)):
                    self.bias_layers[_].eval()
                if inc_i > 0:
                    self.stage1_distill(train_data, criterion, optimizer)
                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
            if inc_i > 0:
                for epoch in range(epoches):
                    # bias_scheduler.step()
                    self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 1 == 0:
                        acc = self.test(test_data)
                        test_acc.append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)
            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print("test_accs")
            print(test_accs)
        return test_accs

    def bias_forward(self, input):
        in1 = input[:, :2]
        in2 = input[:, 2:4]
        in3 = input[:, 4:6]
        in4 = input[:, 6:8]
        in5 = input[:, 8:10]
        
        out1 = self.bias_layer1(in1)
        out2 = self.bias_layer2(in2)
        out3 = self.bias_layer3(in3)
        out4 = self.bias_layer4(in4)
        out5 = self.bias_layer5(in5)
        if self.total_cls == 10:
           return torch.cat([out1, out2, out3, out4, out5], dim = 1)
        elif self.total_cls == 12:
            in6 = input[:, 10:12]
            out6 = self.bias_layer6(in6)
            return torch.cat([out1, out2, out3, out4, out5,out6], dim = 1)
            
        elif elf.total_cls == 14:
            in6 = input[:, 10:12]
            out6 = self.bias_layer6(in6)
            in7 = input[:, 12:14]
            out7 = self.bias_layer7(in7)
        return torch.cat([out1, out2, out3, out4, out5,out6,out7], dim = 1)
        '''elif self.total_cls == 14:
            in6 = input[:, 10:12]
            in7 = input[:, 12:14]
            out6 = self.bias_layer6(in6)
            out7 = self.bias_layer6(in7)
            return torch.cat([out1, out2, out3, out4, out5, out6, out7], dim = 1)
'''

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - 2)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            #if label == -1:
            #    print(label)
            #if label > 10:
            #    print(label)
            #    print("above 10")
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-2]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-2]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))


    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(val_bias_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))
