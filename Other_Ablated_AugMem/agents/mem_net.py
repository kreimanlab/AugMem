'''
Implementation of augmented memory network
'''

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Net(nn.Module):
    def __init__(self, MemNumSlots, MemFeatSz, model_config):
        super(Net, self).__init__()
        
        self.config = model_config
        self.batch_size = 32 #will be overwritten later        
        
        ## parameters for resnet
#         self.compressedChannel = 512
#         self.memsize = 8
#         self.memslots = MemNumSlots
#         self.origsz = 13
#         self.cutlayer = 12
        
        ## parameters for mobilenet
        
        self.memsize = 8
        self.memslots = MemNumSlots               
        
        self.focus_beta = self.config['mem_focus_beta'] #focus on content
        self.sharp_gamma =  1 #focus on locations
        
        ################################ RESNET ######################################
        self.compressedChannel = 512
        self.origsz = 13
        self.cutlayer = 12
        
        # Load pretrained VGG16/resnet Model
        self.model = models.squeezenet1_0(pretrained = self.config['pretrained'])
        self.model.classifier[1] = nn.Conv2d(512,self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))           
        # freezing weights for feature extraction if desired        
        for param in self.model.parameters():
            param.requires_grad = True
        #print(self.model)    
        # Remove last two layers: adaptive pool + fc layers
        self.FeatureExtractor = torch.nn.Sequential(*(list(self.model.features)[:self.cutlayer]))         
        # freezing weights for feature extraction if desired
        if self.config['freeze_feature_extract']:
            for param in self.FeatureExtractor.parameters():
                param.requires_grad = False
                
        self.block = torch.nn.Sequential(*(list(self.model.features)[self.cutlayer:]),
                                         self.model.classifier,
                                        torch.nn.Flatten())
        
        
        ################################ MOBILENET ###################################### 
#         self.compressedChannel = 64
#         self.origsz = 14
#         self.cutlayer = 8
#         def convert_layers(model, layer_type_old):    
#             for name, module in reversed(model._modules.items()):
#                 if len(list(module.children())) > 0:
#                     # recurse
#                     model._modules[name] = convert_layers(module, layer_type_old)

#                 if type(module) == layer_type_old:
#                     layer_old = module
#                     layer_new = torch.nn.Identity() 
#                     model._modules[name] = layer_new

#             return model
    
#         self.model = models.mobilenet_v2(pretrained = self.config['pretrained'])
#         #self.model = convert_layers(self.model, torch.nn.BatchNorm2d)
#         self.model.classifier[1] = nn.Linear(1280,self.config['n_class'])
#         # freezing weights for feature extraction if desired        
#         for param in self.model.parameters():
#             param.requires_grad = True
#         #print(self.model)    
#         self.FeatureExtractor = torch.nn.Sequential(*(list(self.model.features)[:self.cutlayer]))
#         # freezing weights for feature extraction if desired
#         if self.config['freeze_feature_extract']:
#             for param in self.FeatureExtractor.parameters():
#                 param.requires_grad = False
        
#         self.avgpool = torch.nn.Sequential(nn.AvgPool2d(kernel_size=7, stride=1), nn.Flatten()).cuda()
#         self.block = torch.nn.Sequential(*(list(self.model.features)[self.cutlayer:]),
#                                          self.avgpool,
#                                          self.model.classifier)
        #print(self.block)
        ####################################################################################
        
        #print(self.block)
        #self.memory = nn.Linear(self.memsize, self.memsize, False)
        if self.config['mem_pretrain']:
            print('we are using pre-trained memroy weights from cifar100')
            #fpath = './pretrain/cifar100_augmem_pretrain_tesnorkmeans_normalized.pt'
            fpath = './pretrain/cifar100_augmem_pretrain_tesnorkmeans.pt'
            kmeansweights = torch.load(fpath)
            self.memory = Parameter(kmeansweights)
            #self.memory = Parameter(torch.randn(MemNumSlots, self.memsize))
        else:
            self.memory = Parameter(torch.randn(MemNumSlots, self.memsize))
            
        #self.memoryD = Parameter(torch.randn(MemNumSlots, self.memsize))
        if self.config['freeze_memory']:
            self.memory.requires_grad = False
            #self.memoryD.requires_grad = False
        else:
            self.memory.requires_grad = True
            #self.memoryD.requires_grad = True
            
        #self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
    
    
    def forward(self,x):
        self.batch_size = x.size(0)
        
        extracted = self.FeatureExtractor(x)
        #print(extracted.shape)
        #x = extracted.view(-1,512,13,1,13).repeat(1,1,1,self.memslots,1)  
        x = extracted.view(-1,int(self.compressedChannel/self.memsize)*self.memsize,self.origsz,self.origsz) #dim=3
        
        x = x.permute(0,2,3,1)
        #print(x.shape)
        x = x.view(-1,self.origsz,self.origsz,int(self.compressedChannel/self.memsize),self.memsize) #dim=4
        #print(x.shape)
        #self.memory = self.sigmoid(self.memory)
        att_read = self._similarity(x, self.focus_beta, self.memory)
        att_read = self._sharpen(att_read, self.sharp_gamma)        
        #att_read = self.sigmoid(att_read-1)
        #print(att_read[0,0,:])
        #read = F.linear(att_read, self.memory)
        read = att_read.matmul(self.memory)
        
        read = read.view(-1,self.origsz,self.origsz,int(self.compressedChannel/self.memsize)*self.memsize).permute(0,3,1,2)
        read = read.view(-1, self.compressedChannel, self.origsz, self.origsz)
        
        direct = self.block(extracted)        
        out = self.block(read)  
        
        return direct, out, att_read, read, extracted
     
    def forward_woMem(self, x):
        
        self.batch_size = x.size(0)        
        extracted = self.FeatureExtractor(x)
        direct = self.block(extracted)        
        
        return direct       
    
    def forward_attonly(self, read):
        #att_read = att_read.view(-1,13,13,64, self.memslots)        
        #att_read = F.softmax(att_read,dim=3) 
        #read = att_read.matmul(self.memory)
        #read = F.linear(att_read, self.memory)
        read = read.view(-1,self.origsz,self.origsz,int(self.compressedChannel/self.memsize)*self.memsize).permute(0,3,1,2)
        #read = read.view(-1, self.compressedChannel,self.origsz,self.origsz)
        
        out = self.block(read)
        
        return out, read
    
    def forward_directonly(self, extracted):
        extracted = extracted.view(-1,self.compressedChannel, self.origsz, self.origsz)          
        out = self.block(extracted)
        
        return out
    
    def _similarity(self, key, focus_beta, memory):
        #key = key.view(self.batch_size, 1, -1)
        #print(key.shape)
        #print(memory.shape)
        simmat = key.matmul( memory.t())
        #simmat = F.cosine_similarity(memory + 1e-16, key + 1e-16, dim=-1)
        w = F.softmax(focus_beta * simmat, dim=4)
        return w    

    def _sharpen(self, wc, sharp_gamma):
        w = wc ** sharp_gamma
        #print(w.shape)
        w = torch.div(w, torch.sum(w, dim=4).unsqueeze(4)+ 1e-16)
        return w    
        
    def evalModeOn(self):
        self.eval()
        self.FeatureExtractor.eval()
        self.block.eval()
        self.memory.requires_grad = False
        #self.memory.eval()
    
    def trainModeOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        if self.config['freeze_feature_extract']:
            for param in self.FeatureExtractor.parameters():
                param.requires_grad = False
       
        self.memory.requires_grad = False
        
            
    def trainMemoryOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        for param in self.FeatureExtractor.parameters():
            param.requires_grad = False
            
        for param in self.block.parameters():
            param.requires_grad = True
       
        self.memory.requires_grad = True
        
    def trainOverallOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        if self.config['freeze_feature_extract']:
            for param in self.FeatureExtractor.parameters():
                param.requires_grad = False
            
        for param in self.block.parameters():
            param.requires_grad = True
       
        if self.config['freeze_memory']:
            self.memory.requires_grad = False
            #self.memoryD.requires_grad = False
        else:
            self.memory.requires_grad = True
                


