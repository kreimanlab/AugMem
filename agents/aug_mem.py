'''
Implementation of augmented memory based strategies to combat catastrophic forgetting
'''

import numpy as np
import torch
import torch.nn as nn
from .mem_net import Net
from utils.metric import accuracy, AverageMeter, Timer
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import math

class AugMem(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        Parameters
        ----------
        agent_config : {
            lr = float, momentum = float, weight_decay = float,
            model_weights = str,
            gpuid = [int]
            }
        '''
        super(AugMem, self).__init__()
        
        self.config = agent_config
        
        # print function
        self.log = print
        
        # define memory
        self.MemNumSlots = agent_config['memory_Nslots']
        self.MemFeatSz = agent_config['memory_Nfeat']
        self.memory = torch.randn(self.MemNumSlots, self.MemFeatSz) #container for visualization only
        self.capacity = agent_config['memory_size']
        self.MemtopK = agent_config['memory_topK']
        
        # create the model
        self.net = self.create_model()
        
        # define the loss function
        self.criterion_fn = nn.CrossEntropyLoss()
        self.criterion_bn = nn.BCELoss(reduction = 'mean')
        
        self.ListUsedMem = torch.zeros(self.MemNumSlots)

        if self.config['visualize']:
            # for visualization purpose
            # store some examples of reading attention
            self.viz_read_att = {i: [] for i in range(agent_config['n_class'])}
            self.viz_NumEgs = 50 #number of examples to store for visualization
        
        # gpu config
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True            
            self.StorageAttRead = {} #dictionary: storing reading attentions
        else:
            self.gpu = False
            self.StorageAttRead = {} #dictionary: storing reading attentions
        
        # initialize the optimizer
        self.init_optimizer()
        
        # denotes which output nodes are active for each task
        self.active_out_nodes = list(range(agent_config['n_class']))
        
    # creates desired model   
    def create_model(self):
        
        cfg = self.config
        
        # Define the backbone (MLP, LeNet, VGG, ResNet, etc)
        # Model type: MLP, ResNet, etc
        # Model name: MLP100, MLP1000, etc
        # We used modified backbones because of memory, reading, and writing head
        net = Net(self.MemNumSlots, self.MemFeatSz, cfg)
        
        # load pretrained weights if specified
        #eg. cfg['model_weights'] = 'model_pretrained = './pretrained_model/'
        #print('model_weights')
        if cfg['model_weights'] is not None:
            print('=> Load model weights: '+  cfg['model_weights'])
            # loading weights to CPU
            netPath = cfg['model_weights'] + '_net.pth'  
            preweights = torch.load(netPath)
            #print(preweights)
            net.load_state_dict(preweights)            
            print("=> Load done") 
            
        if cfg['memory_weights'] is not None:
            print('=> Load memory weights: '+  cfg['memory_weights'])
            # loading weights to CPU
            netPath = cfg['memory_weights'] + '_mem.pth'  
            preweights = torch.load(netPath)
            #print(preweights)
            net.memory = preweights            
            print("=> Load done") 
        
        return net
    
    # initialize optimzer
    def init_optimizer(self):
        optimizer_arg_net = {'params': self.net.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}        
        
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg_net['momentum'] = self.config['momentum']
            
        elif self.config['optimizer']  in ['Rprop']:
            optimizer_arg_net.pop['weight_decay']
            
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg_net['amsgrad'] == True            
            self.config['optimizer'] = 'Adam'
            
        self.optimizer_net = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg_net)
        
        
    # forward pass of both networks
    def forward(self, x):
        direct, out, att_read, read, extracted = self.net.forward(x)                 
        return direct, out, att_read, read, extracted   
    
    def forward_attonly(self, att_read):
        out, read = self.net.forward_attonly(att_read)
        return out, read
    
    def forward_direct(self, extracted):
        out = self.net.forward_directonly(extracted)
        return out
    
    def forward_woMem(self, x):
        out = self.net.forward_woMem(x)
        return out
    
    # make a prediction
    def predict(self, inputs):
        direct, out, att_read, read, extracted = self.forward(inputs)
        return direct.detach().cpu(), out.detach().cpu(), att_read.view(inputs.size(0),-1).detach().cpu(), read.detach().cpu() 
    
    # calculate loss
    def criterion_classi(self, pred, target):
        # mask inactive output nodes
        pred = pred[:,self.active_out_nodes]
        loss = self.criterion_fn(pred, target)
        return loss
    
    def criterion_BCElogits(self, direct, out):
        # mask inactive output nodes
        direct = direct[:,self.active_out_nodes]
        out = out[:,self.active_out_nodes]
        direct = direct.clone().detach()
        loss = self.criterion_bn(out, direct)
        return loss

    # calculate regularization loss on writing attention on seen classes
    def criterion_regularize(self, memory):
        #print(self.ListUsedMem)
        if self.gpu:
            diff = (memory - self.memory.cuda())**2
            ListUsedMem = self.ListUsedMem.clone().cuda()
        else:
            diff = (memory - self.memory)**2
            ListUsedMem = self.ListUsedMem.clone()
        
        diff = ListUsedMem.view(1,self.MemNumSlots).matmul(diff)
        #print(diff)
        reg_loss = diff.sum()
        loss = self.config['reg_coef']*reg_loss
        return loss   
#    
#    # replay old reading attention 
    def criterion_replay(self, target):
        labelslist = target.tolist()
        #print(self.active_out_nodes)
        replay_labels = [cls for cls in self.active_out_nodes 
                         if cls not in labelslist and cls in self.StorageAttRead.keys()]
        replay_batchsize = len(replay_labels)         
        
        if replay_batchsize > 0:
            
            replaytimes = math.ceil(len(labelslist)/replay_batchsize)
            ReplayAttRead = []
            #print('there are ' + str(replay_batchsize) + ' replays')
            # find corresponding read attention for replaying classes
            for i in range(replaytimes):
                tempatt = [self.StorageAttRead[i][random.randint(0, self.StorageAttRead[i].size(0)-1)] for i in replay_labels[0:replay_batchsize]]
                ReplayAttRead += tempatt
                replay_labels += replay_labels[0:replay_batchsize]
                
            replay_labels = replay_labels[0:-replay_batchsize]
            
            # convert to tensor for training
            ReplayStorage = torch.stack(ReplayAttRead)
            ReplayAttRead = ReplayStorage[:, :int(512/self.MemFeatSz)*13*13*self.MemtopK]
            ReplayAttRead = ReplayAttRead.long()
            #ReplayAttMem = ReplayStorage[:,64*13*13*self.MemtopK:-self.config['n_class']].reshape(len(replay_labels), self.MemNumSlots, self.MemFeatSz)  
            ReplayAttMem = self.memory
            ReplayLogits = ReplayStorage[:,-self.config['n_class']:]
            Reconstructed = []
            for i in range(len(replay_labels)):
                Reconstructed.append(ReplayAttMem[ReplayAttRead[i,:], :])
            ReplayRead = torch.stack(Reconstructed)
            #print(ReplayRead.shape)
            replay_labels = torch.LongTensor(replay_labels)
            if self.gpu:
                replay_labels = replay_labels.cuda()
                ReplayRead = ReplayRead.cuda()
                ReplayLogits = ReplayLogits.cuda()
                
            # replay training starts
            self.optimizer_net.zero_grad()
            ReplayRead = ReplayRead.view(replay_labels.size(0), -1)
            #print(ReplayRead.shape)
            outputs, read = self.forward_attonly(ReplayRead)
            #outputs = self.forward_direct(ReplayAttRead)
            replayloss = self.config['replay_coef']*(self.criterion_classi(outputs, replay_labels) + self.config['logit_coef']*self.criterion_BCElogits(torch.sigmoid(outputs), torch.sigmoid(ReplayLogits)))
            replayloss.backward()
            self.clip_grads(self.net)
            self.optimizer_net.step() 
            
            return replay_batchsize, replayloss
        else:
            return 0, float('NaN')
    
        
    # update storage for reading attention and least memory usage indices based after each epoch
    def updateStorage_epoch(self, train_loader):
        self.net.evalModeOn()       
        print('============================================')
        att_read_dict = {cls: [] for cls in self.active_out_nodes}
        avgSampleNum = math.floor(self.capacity/len(self.active_out_nodes))
        
        #mem = self.memory.clone().view(self.MemNumSlots*self.MemFeatSz)        
        # iterating over train loader
        for i, (inputs, target) in enumerate(train_loader):
        
            # transferring to gpu if applicable
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()
                
            # we store/update our storage Reading Attention
            direct, out, att_read, read, extracted = self.forward(inputs)
            #att_read = extracted.detach().view(inputs.size(0),-1)
            att_read = att_read.detach().cpu().view(-1, 13, 13, int(512/self.MemFeatSz), self.MemNumSlots)
            #print(att_read.shape)
            #find top retireved memory address
            att_read_ind = torch.argsort(att_read, dim=4, descending=True)
            #print(torch.max(att_read_ind))
            att_read_ind = att_read_ind[:,:,:,:,0:self.MemtopK]
            att_read = att_read_ind.view(inputs.size(0),-1)
            att_read = att_read.view(inputs.size(0),-1)
            
            direct = torch.reshape(direct.detach().cpu(), (inputs.size(0), self.config['n_class']))
            for cls in self.active_out_nodes: 

                #allAttRead = [torch.cat((att_read[i], mem, direct[i]), 0) for i in range(att_read.size(0)) if target[i] == cls ]
                allAttRead = [torch.cat((att_read[i], direct[i]), 0) for i in range(att_read.size(0)) if target[i] == cls ]
                #select = range(len(allAttRead))
                att_read_dict[cls] =  att_read_dict[cls] + allAttRead[0::5] 
                nowSampleNum = len(att_read_dict[cls])
                if nowSampleNum > avgSampleNum:
                    #serve as a queue to pop out old ones
                    att_read_dict[cls] = att_read_dict[cls][nowSampleNum - avgSampleNum:]
                
        for cls in self.active_out_nodes:
            if len(att_read_dict[cls])!= 0:
                self.StorageAttRead[cls] = torch.stack(att_read_dict[cls], dim=0)#.mean(0)
                #print(self.StorageAttRead[cls].shape)
                print('storing class id: ' + str(cls) + ' for num of samples = ' + str(self.StorageAttRead[cls].size(0)))
                
        # update least used memory slots
        for cls in range(self.config['n_class']):
            if cls in self.StorageAttRead.keys():
                #print(self.StorageAttRead[cls][:,:13*13*64*self.MemtopK].shape)
                sz_w = self.StorageAttRead[cls][:,:13*13*int(512/self.MemFeatSz)*self.MemtopK].size(0)
                sz_h = self.StorageAttRead[cls][:,:13*13*int(512/self.MemFeatSz)*self.MemtopK].size(1)
                indices = torch.reshape(self.StorageAttRead[cls][:, :13*13*int(512/self.MemFeatSz)*self.MemtopK], (sz_w, sz_h)).long()
                #print(indices.shape)
                #print(torch.max(indices))
                self.ListUsedMem[indices] = 1
        self.memory = self.net.memory.clone().detach().cpu()          
            
        self.net.trainModeOn()

    # update storage for reading attention and least memory usage indices based after each epoch
    # This is an alternative version of the function that uses the herding strategy
    # ONLY USED FOR ABLATION STUDIES as of Nov 20, 2021.
    def update_HERDING_Storage_epoch(self, train_loader):
        self.net.evalModeOn()
        print('============================================')
        att_read_dict = {cls: [] for cls in self.active_out_nodes}
        avgSampleNum = math.floor(self.capacity / len(self.active_out_nodes))
        features = []

        # mem = self.memory.clone().view(self.MemNumSlots*self.MemFeatSz)
        # iterating over train loader
        for i, (inputs, target) in enumerate(train_loader):

            # transferring to gpu if applicable
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            # we store/update our storage Reading Attention
            direct, out, att_read, read, extracted = self.forward(inputs)
            # att_read = extracted.detach().view(inputs.size(0),-1)
            att_read = att_read.detach().cpu().view(-1, self.compressedWidth, self.compressedWidth,
                                                    int(self.compressedChannel / self.memsize), self.MemNumSlots)
            # find top retireved memory address
            att_read_ind = torch.argsort(att_read, dim=4, descending=True)
            # print(torch.max(att_read_ind))
            att_read_ind = att_read_ind[:, :, :, :, 0:self.MemtopK]
            att_read = att_read_ind.view(inputs.size(0), -1)
            att_read = att_read.view(inputs.size(0), -1)

            read = read.detach().cpu().reshape(inputs.size(0),
                                               self.compressedWidth * self.compressedWidth * self.compressedChannel)

            direct = torch.reshape(direct.detach().cpu(), (inputs.size(0), self.config['n_class']))

            target = target.detach().cpu()

            # print(lab.shape)
            # print(read[i].shape, att_read[i].shape, direct[i].shape, lab[0].shape)
            for i in range(att_read.size(0)):
                lab = torch.zeros(1, 1)
                lab[0, 0] = target[i].item()
                feat = torch.cat((read[i], att_read[i], direct[i], lab[0]), 0).unsqueeze(0)
                # print(feat.shape)
                features.append(feat)

                # print('==========================================')

        # get features
        features = torch.cat(features, 0)

        # normalize features
        feature_sz_read = self.compressedWidth * self.compressedWidth * self.compressedChannel
        for i in range(features.shape[0]):
            features[i, :feature_sz_read] = features[i, :feature_sz_read] / features[i, :feature_sz_read].norm()

            # getting mean & normalizing
        features = features.numpy()
        labelist = features[:, -1]
        uni_class_list = np.unique(labelist)
        # print(features.shape)

        for ucls in uni_class_list:

            feat_cls_ind = np.where(labelist == ucls)
            # print('feat_cls_ind',feat_cls_ind[0].shape)
            class_mean = np.mean(features[feat_cls_ind[0], :feature_sz_read], axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean)
            # print(class_mean.shape)

            # select examplar set
            exemplar_set = []
            # list of tensors of shape (feature_size,)
            exemplar_features = []
            trackid = []
            # computing exemplars
            for k in range(avgSampleNum):
                S = np.sum(exemplar_features, axis=0)
                phi = features[feat_cls_ind[0], :feature_sz_read]
                mu = class_mean
                mu_p = (1.0 / (k + 1)) * (phi + S)
                # normalize
                mu_p = mu_p / np.linalg.norm(mu_p)
                # print('mu', mu.shape)
                # print('mu_p', mu_p.shape)
                diff = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                # print(diff.shape)
                # e = np.argmin(diff)
                e = np.argsort(diff)
                # print(e)
                for ind in e:
                    if ind not in trackid:
                        # print(ind)
                        trackid.append(ind)
                        exemplar_set.append(torch.Tensor(features[feat_cls_ind[0][ind], feature_sz_read:-1]))
                        exemplar_features.append(features[feat_cls_ind[0][ind], :feature_sz_read])
                        break

                # features = np.delete(features, e, axis = 0)

            # print(exemplar_set[0].shape)
            exemplars = torch.stack(exemplar_set, dim=0)
            # print(exemplars.shape)
            self.StorageAttRead[int(ucls)] = exemplars
            print('storing class id: ' + str(int(ucls)) + ' for num of samples = ' + str(
                self.StorageAttRead[ucls].size(0)))

        # removing extra samples from previous tasks
        for cls in self.active_out_nodes:
            self.StorageAttRead[cls] = self.StorageAttRead[cls][:avgSampleNum]

        # update least used memory slots
        for cls in range(self.config['n_class']):
            if cls in self.StorageAttRead.keys():
                # print(self.StorageAttRead[cls][:,:13*13*64*self.MemtopK].shape)
                sz_w = self.StorageAttRead[cls][:, :self.compressedWidth * self.compressedWidth * int(
                    self.compressedChannel / self.memsize) * self.MemtopK].size(0)
                sz_h = self.StorageAttRead[cls][:, :self.compressedWidth * self.compressedWidth * int(
                    self.compressedChannel / self.memsize) * self.MemtopK].size(1)
                indices = torch.reshape(self.StorageAttRead[cls][:,
                                        :self.compressedWidth * self.compressedWidth * int(
                                            self.compressedChannel / self.memsize) * self.MemtopK],
                                        (sz_w, sz_h)).long()
                # print(indices.shape)
                # print(torch.max(indices))
                self.ListUsedMem[indices] = 1

        self.memory = self.net.memory.clone().detach().cpu()

        self.net.trainModeOn()

    # compute validation loss/accuracy (being called outside class)
    def validation(self, dataloader):
        acc_out = AverageMeter()
        acc_dir = AverageMeter()
        batch_timer = Timer()
        batch_timer.tic()
        
        self.net.evalModeOn()        
        # keeping track of prior mode
        if self.config['visualize']:
            self.viz_read_att = {i: [] for i in range(self.config['n_class'])}
            self.viz_input = {i: [] for i in range(self.config['n_class'])}
            self.viz_direct = {i: [] for i in range(self.config['n_class'])}
            self.confusemat = torch.zeros((len(self.active_out_nodes),len(self.active_out_nodes)))
        
        for i, (inputs, target) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()
             
            direct, output, att_read, read = self.predict(inputs)
            
            output = output[:,self.active_out_nodes]
            direct = direct[:,self.active_out_nodes]
            
            if self.gpu:
                target = target.cpu()
            acc_out.update(accuracy(output, target), inputs.size(0))
            acc_dir.update(accuracy(direct, target), inputs.size(0))

            if self.config['visualize']:
                #save some examples for visualization
                for bat in range(att_read.size(0)):
                    label = target[bat].item()

                    #save confusion mat
                    direct_vec = direct[bat,:].view(1,-1)
                    _, predicted_label = direct_vec.topk(k = 1, dim = 1, largest = True, sorted = True)
                    self.confusemat[label, predicted_label] = self.confusemat[label, predicted_label] + 1

                    if len(self.viz_read_att[label]) < self.viz_NumEgs:
                        #view(-1,13,13,64,100) as shape for attention read
                        att_read_bat = att_read[bat,:].view(1, 13, 13, int(512/self.MemFeatSz), self.MemNumSlots)
                        #find top retireved memory address
                        att_read_ind = torch.argsort(att_read_bat, dim=4, descending=True)
                        att_read_ind = att_read_ind[:,:,:,:,0:self.MemtopK]
                        att_read_bat = att_read_ind.view(1, 13, 13, int(512/self.MemFeatSz), self.MemtopK)
                        att_read_np = att_read_bat.numpy()
                        self.viz_read_att[label].append(att_read_np)
                        self.viz_direct[label].append(direct_vec.detach().cpu().numpy())
                        self.viz_input[label].append(inputs[bat,:,:,:].view(1,3,224,224).detach().cpu().numpy())
        # return model to original mode
        #self.net.trainModeOn()
        #self.classinet.trainModeOn()
        total_time = batch_timer.toc()
        
        return acc_out.avg, acc_dir.avg, total_time
        
        
    # stream learning (being called outside class)
    def learn_stream(self, train_loader, task):        
        
        self.net.trainOverallOn()
        losses_classidir = AverageMeter()
        losses_classiout = AverageMeter()
        losses_logits = AverageMeter()
        losses_reg = AverageMeter()
        losses = AverageMeter()
        losses_replay = AverageMeter()
        acc_out = AverageMeter()
        acc_dir = AverageMeter()
        
        data_timer = Timer()
        batch_timer = Timer()
        forward_timer = Timer()
        backward_timer = Timer()
        
        data_time = AverageMeter()
        batch_time = AverageMeter()
        forward_time = AverageMeter()
        backward_time = AverageMeter()
        
        self.log('Batch\t Loss\t\t Acc')
        data_timer.tic()
        batch_timer.tic()
        
        # iterating over train loader
        for i, (inputs, target) in enumerate(train_loader):             
            # transferring to gpu if applicable
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()
            
            # measure data loading time
            data_time.update(data_timer.toc())           
            self.optimizer_net.zero_grad()
            
            ## FORWARD pass
            # getting loss, updating model
            forward_timer.tic()
            #direct, output, att_read, read = self.forward(inputs)
            direct, output, att_read, read, extracted = self.forward(inputs)
            forward_time.update(forward_timer.toc())
            
            ## BACKWARD pass
            backward_timer.tic()
            #classification loss
            loss_classiout = self.criterion_classi(output, target)
            loss_classidir = self.criterion_classi(direct, target) 
            logitsloss = self.config['logit_coef']*self.criterion_BCElogits(torch.sigmoid(output), torch.sigmoid(direct))
            loss_reg = self.criterion_regularize(self.net.memory)
            loss = loss_classiout + loss_classidir + logitsloss + loss_reg           
            
            loss.backward()
            self.clip_grads(self.net)
            self.optimizer_net.step()                 
            ## IMPORTANT: have to detach memory for next step; otherwise, backprop errors
            #update augmented memory
            self.memory = self.net.memory.clone().detach().cpu()        
            backward_time.update(backward_timer.toc())

            if task > 0:
                ## REPLAY read content from classes not belonging to target labels from memory
                for time in range(self.config['replay_times']):
                    replay_size, replay_loss = self.criterion_replay(target)
                #replay_size = 0
                #replay_loss = None
                if self.config['replay_times'] > 0 and replay_size > 0:
                    losses_replay.update(replay_loss, replay_size)
                else:
                    replay_size = 0
                    replay_loss = None
                
            ## UPDATE storage of reading attention
            #self.updateStorage_batch(inputs, target)
            
            ## COMPUTE accuracy            
            inputs = inputs.detach()
            target = target.detach()            
            # mask inactive output nodes
            direct = direct[:,self.active_out_nodes]
            output = output[:,self.active_out_nodes] 
            
            acc_out.update(accuracy(output, target), inputs.size(0))
            acc_dir.update(accuracy(direct, target), inputs.size(0))
            losses.update(loss, inputs.size(0))
            losses_classiout.update(loss_classiout, inputs.size(0))
            losses_classidir.update(loss_classidir, inputs.size(0))
            losses_logits.update(logitsloss, inputs.size(0))
            losses_reg.update(loss_reg, inputs.size(0))
            
            # measure elapsed time for entire batch
            batch_time.update(batch_timer.toc())
            # updating these timers with with current time
            data_timer.toc()
            forward_timer.toc()
            backward_timer.toc()
            
            self.log('[{0}/{1}]\t'
                          'L = {loss.val:.3f} ({loss.avg:.3f})\t'
                          'L-dir = {losses_classidir.val:.3f} ({losses_classidir.avg:.3f})\t'
                          'L-out = {losses_classiout.val:.3f} ({losses_classiout.avg:.3f})\t'
                          'L-log = {losses_logits.val:.3f} ({losses_logits.avg:.3f})\t'
                          'L-reg = {losses_reg.val:.3f} ({losses_reg.avg:.3f})\t'
                          'L-rep = {loss_replay.val:.3f} ({loss_replay.avg:.3f})\t'
                          'A-out = {acc_out.val:.2f} ({acc_out.avg:.2f})\t'
                          'A-dir = {acc_dir.val:.2f} ({acc_dir.avg:.2f})'.format(
                        i, len(train_loader), loss=losses, losses_classidir = losses_classidir, losses_classiout=losses_classiout, losses_logits = losses_logits,losses_reg = losses_reg, loss_replay=losses_replay, acc_out=acc_out, acc_dir=acc_dir))
            # self.log('{batch_time.val:3f}\t'
            #               '{data_time.val:3f}\t'
            #               '{forward_time.val:3f}\t {backward_time.val:3f}'.format(
            #             batch_time=batch_time, data_time=data_time, forward_time=forward_time, backward_time=backward_time))

        self.log(' * Train Acc: A-out = {acc_out.avg:.3f} \t A-dir = {acc_dir.avg:.3f}'.format(acc_out=acc_out, acc_dir=acc_dir))
        self.log(' * Avg. Data time: {data_time.avg:.3f}, Avg. Batch time: {batch_time.avg:.3f}, Avg. Forward time: {forward_time.avg:.3f}, Avg. Backward time: {backward_time.avg:.3f}'
                 .format(data_time=data_time, batch_time=batch_time, forward_time=forward_time, backward_time=backward_time))

        ## END of epoch train data
        ## UPDATE storage of reading attention
        ## UPDATE least used memory slot indices
        if self.config['herding_mode']:
            # This will only occur during the "herding" ablation study.
            self.update_HERDING_Storage_epoch(train_loader)
        else:
            self.updateStorage_epoch(train_loader)
        
        if task != 0:
            for g in self.optimizer_net.param_groups:
                g['lr'] = 0.001                
        

#        print('self.StorageAttRead')
#        print(self.StorageAttRead)
#        print('self.StorageAttWrite')
#        print(self.StorageAttWrite)
#        print('self.ListUsedMem')
#        print(self.ListUsedMem)
        
        
    # overriding cuda function
    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.net = self.net.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        
        return self
    
    # save current model state
    def save_model(self, filename):
        net_state = self.net.state_dict()        
        netPath = filename + '_net.pth'        
        print('=> Saving models and aug memory to:', filename)
        torch.save(net_state, netPath)
        print('=> Save Done')
        
    def save_memory(self, filename):
        net_state = self.net.memory      
        netPath = filename + '_mem.pth'        
        print('=> Saving models and aug memory to:', filename)
        torch.save(net_state, netPath)
        print('=> Save Done')

    def clip_grads(self, model):
        """Gradient clipping to the range [10, 10]."""
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)    
        
    def visualize_att_read(self, filename):
        for i in self.active_out_nodes: #range(self.config['n_class']):
            att = self.viz_read_att[i]
            arr = np.vstack(att)   
            
            inputs = self.viz_input[i]
            arr_inputs = np.vstack(inputs)
            #sort in descending order and return top 5 index
            #print((-np.mean(arr, axis=0)).argsort(axis=0))
            
            direct = self.viz_direct[i]
            arr_direct = np.vstack(direct)
            
            sio.savemat(filename + '_att_read_' + str(i) + '.mat', {'att_read':arr,'inputs':arr_inputs, 'direct':arr_direct})
        sio.savemat(filename + '_confusemat.mat', {'confusemat': self.confusemat.numpy()})    
        
            #b = plt.imshow(arr[:self.viz_NumEgs,:], cmap='hot')
            #plt.colorbar(b)
            #plt.show()
        
    def visualize_memory(self, filename):
        plotmem = self.memory.clone()
        sio.savemat(filename + '_memory.mat', {'memory':plotmem.detach().cpu().numpy()})
        #b = plt.imshow(plotmem.detach().cpu().numpy(), cmap='hot')
        #plt.colorbar(b)
        #plt.show()
        