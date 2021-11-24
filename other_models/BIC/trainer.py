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
from cifar100 import cifar100
from exemplar import Exemplar
from copy import deepcopy


class Trainer:
    def __init__(self, total_cls, paradigm, run,dataset):
        self.total_cls = total_cls
        self.seen_cls = 0
        #self.dataset = Cifar100()
        if dataset == "core50":
            self.dataset = Core50(paradigm, run)
        if dataset == 'toybox':
            self.dataset = Toybox(paradigm, run)
        if dataset == "ilab":
            print("in ilab data")
            self.dataset = Ilab(paradigm, run)
        if dataset == "cifar100":
            print("in cifar100 data")
            self.dataset = cifar100(paradigm, run)
        
        print("total_cls is")
        print(total_cls)
        self.model = PreResNet(32,total_cls).cuda()
        print(self.model)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.bias_layer5 = BiasLayer().cuda()
        # if self.total_cls == 10:
        #     self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5]

        if self.total_cls == 12:
            self.bias_layer6 = BiasLayer().cuda()
            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,self.bias_layer6]

        if self.total_cls == 14:
            print("for ilab data")
            self.bias_layer6 = BiasLayer().cuda()
            self.bias_layer7 = BiasLayer().cuda()
            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,self.bias_layer6, self.bias_layer7]
            
        if self.total_cls == 100:
            print("for ilab data")
            self.bias_layer6 = BiasLayer().cuda()
            self.bias_layer7 = BiasLayer().cuda()
            self.bias_layer8 = BiasLayer().cuda()
            self.bias_layer9 = BiasLayer().cuda()
            self.bias_layer10 = BiasLayer().cuda()
            self.bias_layer11 = BiasLayer().cuda()
            self.bias_layer12 = BiasLayer().cuda()
            self.bias_layer13 = BiasLayer().cuda()
            self.bias_layer14 = BiasLayer().cuda()
            self.bias_layer15 = BiasLayer().cuda()
            self.bias_layer16 = BiasLayer().cuda()
            self.bias_layer17 = BiasLayer().cuda()
            self.bias_layer18 = BiasLayer().cuda()
            self.bias_layer19 = BiasLayer().cuda()
            self.bias_layer20 = BiasLayer().cuda()

            self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5,self.bias_layer6, self.bias_layer7,
                                self.bias_layer8,self.bias_layer9,self.bias_layer10,self.bias_layer11,self.bias_layer12,self.bias_layer13,self.bias_layer14,
                                self.bias_layer15,self.bias_layer16,self.bias_layer17,self.bias_layer18,self.bias_layer19,self.bias_layer20]
            
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
        task8,task9,task10,task11,task12,task13,task14 = 0,0,0,0,0,0,0
        task15,task16,task17,task18,task19,task20 = 0,0,0,0,0,0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0
        sum8,sum9,sum10,sum11,sum12,sum13,sum14 = 0,0,0,0,0,0,0
        sum15,sum16,sum17,sum18,sum19,sum20 = 0,0,0,0,0,0


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
                    if a >= 0 and a <= 4:
                        task1 += 1
                    elif a >= 5 and a <= 9:
                        task2 += 1
                    elif a >= 10 and a <= 14:
                        task3 += 1
                    elif a >= 15 and a <= 19:
                        task4 += 1
                    elif a >= 20 and a <= 24:
                        task5 += 1
                    elif a >= 25 and a <= 29:
                        task6 += 1
                    elif a >= 30 and a <= 34:
                        task7 += 1
                    elif a >= 35 and a <= 39:
                        task8 += 1
                    elif a >= 40 and a <= 44:
                        task9 += 1
                    elif a >= 45 and a <= 49:
                        task10 += 1
                    elif a >= 50 and a <= 54:
                        task11 += 1
                    elif a >= 55 and a <= 59:
                        task12 += 1
                    elif a >= 60 and a <= 64:
                        task13 += 1
                    elif a >= 65 and a <= 69:
                        task14 += 1
                    elif a >= 70 and a <= 74:
                        task15 += 1
                    elif a >= 75 and a <= 79:
                        task16 += 1
                    elif a >= 80 and a <= 84:
                        task17 += 1
                    elif a >= 85 and a <= 89:
                        task18 += 1
                    elif a >= 90 and a <= 94:
                        task19 += 1
                    elif a >= 95 and a <= 99:
                        task20 += 1

                if a >= 0 and a <= 4:
                    sum1 += 1
                elif a >= 5 and a <= 9:
                    sum2 += 1
                elif a >= 10 and a <= 14:
                    sum3 += 1
                elif a >= 15 and a <= 19:
                    sum4 += 1
                elif a >= 20 and a <= 24:
                    sum5 += 1
                elif a >= 25 and a <= 29:
                    sum6 += 1
                elif a >= 30 and a <= 34:
                    sum7 += 1
                elif a >= 35 and a <= 39:
                    sum8 += 1
                elif a >= 40 and a <= 44:
                    sum9 += 1
                elif a >= 45 and a <= 49:
                    sum10 += 1
                elif a >= 50 and a <= 54:
                    sum11 += 1
                elif a >= 55 and a <= 59:
                    sum12 += 1
                elif a >= 60 and a <= 64:
                    sum13 += 1
                elif a >= 65 and a <= 69:
                    sum14 += 1
                elif a >= 70 and a <= 74:
                    sum15 += 1
                elif a >= 75 and a <= 79:
                    sum16 += 1
                elif a >= 80 and a <= 84:
                    sum17 += 1
                elif a >= 85 and a <= 89:
                    sum18 += 1
                elif a >= 90 and a <= 94:
                    sum19 += 1
                elif a >= 95 and a <= 99:
                    sum20 += 1
    

        # print("pred and label")
        # print(pred,label)

        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        # print("task wise")
        # print(task1, task2, task3, task4, task5,task6, task7)
        # print("task wise sum")
        # print(sum1, sum2, sum3, sum4,sum5, sum6, sum7) 
        print("Task wise accuracies:")
        if sum1 != 0:
            #task1_res.append(float(task1 / sum1))
            acc_1sttask = task1 / sum1
            print("task 1:",task1 / sum1)
        if sum2 != 0:
            print("task 2:",task2 / sum2)
        if sum3 != 0:
            print("task 3:",task3 / sum3)
        if sum4 != 0:
            print("task 4:",task4 / sum4)
        if sum5 != 0:
            print("task 5:",task5 / sum5)
        if sum6 != 0:
            print("task 6:",task6 / sum6)
        if sum7 != 0:
            print("task 7:",task7 / sum7)
        if sum8 != 0:
            print("task 8:",task8 / sum8)
        if sum9 != 0:
            print("task 9:",task9 / sum9)
        if sum10 != 0:
            print("task 10:",task10 / sum10)
        if sum11 != 0:
            print("task 11:",task11 / sum11)
        if sum12 != 0:
            print("task 12:",task12 / sum12)
        if sum13 != 0:
            print("task 13:",task13 / sum13)
        if sum14 != 0:
            print("task 14:",task14 / sum14)
        if sum15 != 0:
            print("task 15:",task15 / sum15)
        if sum16 != 0:
            print("task 16:",task16 / sum16)
        if sum17 != 0:
            print("task 17:",task17 / sum17)
        if sum18 != 0:
            print("task 18:",task18 / sum18)
        if sum19 != 0:
            print("task 19:",task19 / sum19)
        if sum20 != 0:
            print("task 20:",task20 / sum20)

        # print("correct")
        # print(correct)
        # print("wrong + correct")
        # print(wrong + correct)
        self.model.train()
        print("---------------------------------------------")
        return acc,acc_1sttask


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
        first_task_test_res_final = []
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
            #exemplar.update(total_cls//dataset.batch_num, (train_x, train_y), (val_x, val_y)) #is this even correct????? #RBZ changes
            exemplar.update(total_cls//20, (train_x, train_y), (val_x, val_y))

            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform),
                        batch_size=1, shuffle=True, drop_last=False)
            test_acc = []
            first_task_test_res = []



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
                acc,_ = self.test(test_data)
            if inc_i > 0:
                for epoch in range(epoches):
                    # bias_scheduler.step()
                    self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 1 == 0:
                        acc,_ = self.test(test_data)
                        test_acc.append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)
            self.previous_model = deepcopy(self.model)
            acc,acc_1stTask = self.test(test_data)
            test_acc.append(acc)
            first_task_test_res.append(acc_1stTask)

            test_accs.append(max(test_acc))
            first_task_test_res_final.append(max(first_task_test_res)) #probably doesnt matter because of 1epoch afterwards

            print("test_accs")
            print(test_accs)
        return test_accs,first_task_test_res_final

    def bias_forward(self, input):
        in1 = input[:, :5]
        in2 = input[:, 5:10]
        in3 = input[:, 10:15]
        in4 = input[:, 15:20]
        in5 = input[:, 20:25]
        in6 = input[:, 25:30]
        in7 = input[:, 30:35]
        in8 = input[:, 35:40]
        in9 = input[:, 40:45]
        in10 = input[:, 45:50]
        in11 = input[:, 50:55]
        in12 = input[:, 55:60]
        in13 = input[:, 60:65]
        in14 = input[:, 65:70]
        in15 = input[:, 70:75]
        in16 = input[:, 75:80]
        in17 = input[:, 80:85]
        in18 = input[:, 85:90]
        in19 = input[:, 90:95]
        in20 = input[:, 95:100]
        
        out1 = self.bias_layer1(in1)
        out2 = self.bias_layer2(in2)
        out3 = self.bias_layer3(in3)
        out4 = self.bias_layer4(in4)
        out5 = self.bias_layer5(in5)
        out6,out7,out8,out9 = self.bias_layer6(in6),self.bias_layer7(in7),self.bias_layer8(in8),self.bias_layer9(in9)
        out10,out11,out12,out13 = self.bias_layer10(in10),self.bias_layer11(in11),self.bias_layer12(in12),self.bias_layer13(in13)
        out14,out15,out16,out17 = self.bias_layer14(in14),self.bias_layer15(in15),self.bias_layer16(in16),self.bias_layer17(in17)
        out18,out19,out20 = self.bias_layer18(in18),self.bias_layer19(in19),self.bias_layer20(in20)
        
         
        if self.total_cls == 10:
           return torch.cat([out1, out2, out3, out4, out5], dim = 1)
        elif self.total_cls == 12:
            in6 = input[:, 10:12]
            out6 = self.bias_layer6(in6)
            return torch.cat([out1, out2, out3, out4, out5,out6], dim = 1)
            
        elif self.total_cls == 14:
            in6 = input[:, 10:12]
            out6 = self.bias_layer6(in6)
            in7 = input[:, 12:14]
            out7 = self.bias_layer7(in7)
            return torch.cat([out1, out2, out3, out4, out5,out6,out7], dim = 1)

        if self.total_cls == 100:
            #print("in total_cls = 100")
            return torch.cat([out1, out2, out3, out4, out5,
           out6,out7,out8,out9,out10,out11,out12,out13,
           out14,out15,out16,out17,out18,out19,out20], dim = 1)

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
        T = 5
        alpha = (self.seen_cls - 5)/ self.seen_cls
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
