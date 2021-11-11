'''
defining naive agent with no methods to combat catastrophic forgetting
'''

import torch
import torch.nn as nn
import models
from utils.metric import accuracy, AverageMeter, Timer
import scipy.io as sio
import numpy as np

class NormalNN(nn.Module):
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
        super(NormalNN, self).__init__()
        
        self.config = agent_config
        
        # print function
        self.log = print
        
        # create the model
        self.model = self.create_model()
        
        # define the loss function
        self.criterion_fn = nn.CrossEntropyLoss()
        
        # gpu config
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
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
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](cfg)
        
        # load pretrained weights if specified
        if cfg['model_weights'] is not None:
            print('=> Load model weights: '. cfg['model_weights'])
            # loading weights to CPU
            model_state = torch.load(cfg['model_weights'],
                                     map_location = lambda storage, loc: storage)
            model.load_state_dict(model_state)
            print("=> Load done")
            
        return model
    
    # initialize optimzer
    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer']  in ['Rprop']:
            optimizer_arg.pop['weight_decay']
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] == True
            self.config['optimizer'] = 'Adam'
            
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
    # forward pass of network
    def forward(self, x):
        return self.model.forward(x)
    
    # make a prediction
    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs).detach()
        return (out)
    
    # calculate loss
    def criterion(self, pred, target):
        # mask inactive output nodes
        pred = pred[:,self.active_out_nodes]
        loss = self.criterion_fn(pred, target)
        return loss
    
    # compute validation loss/accuracy
    def validation(self, dataloader):
        
        acc = AverageMeter()
        batch_timer = Timer()
        batch_timer.tic()
        
        self.viz_direct = {i: [] for i in range(self.config['n_class'])}
        self.confusemat = torch.zeros((len(self.active_out_nodes),len(self.active_out_nodes))) 
        self.viz_NumEgs = 50
        
        # keeping track of prior mode
        orig_mode = self.training
        
        self.eval()
                    
        for i, (inputs, target) in enumerate(dataloader):
            
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()
                                                        
            output = self.predict(inputs)
            
            #print('Printing output shape, 0')
            #print(output.shape)
            #print(output[0])
                   
            output = output[:,self.active_out_nodes]
            
            #print('Printing output shape, 1')
            #print(output.shape)
            #print(output[0])

            acc.update(accuracy(output, target), inputs.size(0))
            
            #save some examples for visualization
            for bat in range(output.size(0)):
                label = target[bat].item()   
                
                #save confusion mat
                direct_vec = output[bat,:].view(1,-1) 
                _, predicted_label = direct_vec.topk(k = 1, dim = 1, largest = True, sorted = True)
                self.confusemat[label, predicted_label] = self.confusemat[label, predicted_label] + 1
                
                if len(self.viz_direct[label]) < self.viz_NumEgs:                       
                    self.viz_direct[label].append(direct_vec.detach().cpu().numpy())
                    
        # return model to original mode
        self.train(orig_mode)
        total_time = batch_timer.toc()
        
        return acc.avg, total_time
    
    def visualize_att_read(self, filename):
        for i in self.active_out_nodes: #range(self.config['n_class']):
            
            direct = self.viz_direct[i]
            arr_direct = np.vstack(direct)
            
            sio.savemat(filename + '_att_read_' + str(i) + '.mat', {'direct':arr_direct})
        sio.savemat(filename + '_confusemat.mat', {'confusemat': self.confusemat.numpy()})    
    
    # perform model update
    def update_model(self, out, targets):
        #print(out)
        #print(targets)
        #print(out.shape)
        #print(targets.shape)
        loss = self.criterion(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach() #mengmi
    
    # stream learning
    def learn_stream(self, train_loader, new_task=True):
        
        losses = AverageMeter()
        acc = AverageMeter()
        
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
            
                
            # getting loss, updating model
            forward_timer.tic()
            output = self.forward(inputs)
            forward_time.update(forward_timer.toc())
            
            backward_timer.tic()
            loss = self.update_model(output, target)
            backward_time.update(backward_timer.toc())
            
            inputs = inputs.detach()
            target = target.detach()
            
            # mask inactive output nodes
            output = output[:,self.active_out_nodes]
            
            acc.update(accuracy(output, target), inputs.size(0))
            losses.update(loss, inputs.size(0))
            
            # measure elapsed time for entire batch
            batch_time.update(batch_timer.toc())
            # updating these timers with with current time
            data_timer.toc()
            forward_timer.toc()
            backward_timer.toc()
            
            self.log('[{0}/{1}]\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), loss=losses, acc=acc))
            # self.log('{batch_time.val:3f}\t'
            #               '{data_time.val:3f}\t'
            #               '{forward_time.val:3f}\t {backward_time.val:3f}'.format(
            #             batch_time=batch_time, data_time=data_time, forward_time=forward_time, backward_time=backward_time))

        self.log(' * Train Acc: {acc.avg:.3f}'.format(acc=acc))
        self.log(' * Avg. Data time: {data_time.avg:.3f}, Avg. Batch time: {batch_time.avg:.3f}, Avg. Forward time: {forward_time.avg:.3f}, Avg. Backward time: {backward_time.avg:.3f}'
                 .format(data_time=data_time, batch_time=batch_time, forward_time=forward_time, backward_time=backward_time))


    # overriding cuda function
    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        
        # multi-GPU support
        if len(self.config['gpuid']) > 1:
            # need to change this to DistributedDataParallel, which is preferred
            self.model = nn.parallel.DataParallel(self.model, device_ids = self.config['gpuid'],
                                               output_device = self.config['gpuid'][0])
        return self
    
    # save current model state
    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  
            # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

        
        
        
        
        
        
        