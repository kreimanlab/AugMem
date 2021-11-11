'''
Implementation of regularization based strategies to combat catastrophic forgetting
'''

import torch
import random
from .default import NormalNN

'''
Implementation of L2 regularization
'''
class L2(NormalNN):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, agent_config):
        super(L2, self).__init__(agent_config)
        # for convenience
        self.params = {n: p for n,p in self.model.named_parameters() if p.requires_grad}
        #print (self.params)
        # to store task parameters and importance
        self.regularization_terms = {}
        self.task_count = 0
        # True: there will only be one importance matrix and previous model parameters
        # False: Each task has its own importance matrix and model parameters
        self.online_reg = True
        
        
    def calculate_importance(self, dataloader):
        # use an identity importance so it is L2 regularization
        importance = {}
        for n, p in self.params.items():
            # identity
            importance[n] = p.clone().detach().fill_(1) #.cpu()
        return importance
    
    
    def learn_stream(self, train_loader, new_task_next=True):
        
        print('#reg terms: ', len(self.regularization_terms))
        
        # 1. Learn the parameters for the current task
        super(L2, self).learn_stream(train_loader)
        
        # 2. Back up the weights for the current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()
            
        # 3. Calculate the importance of weights for the current task
        importance = self.calculate_importance(train_loader)
        
        # Save the weight and importance of weights of current task
        if new_task_next:
            self.task_count += 1
            #print(']]]]]]]]]]]]]]]]]]]]]')
            #print(self.task_count)
            if self.online_reg and len(self.regularization_terms) > 0:
                # always use only one slot in regularization_terms
                self.regularization_terms[1] = {'importance': importance,
                                                'task_param': task_param}
            else:
                # use new slot to store task-specific information
                self.regularization_terms[self.task_count] = {'importance': importance,
                                                              'task_param': task_param}
        
    def criterion(self, inputs, targets, regularization = True, **kwargs):
        loss = super(L2, self).criterion(inputs, targets, **kwargs)
        
        # calculate reg_loss only when regularization_terms exists
        if regularization and len(self.regularization_terms)>0:
            reg_loss = 0
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                #print(importance)
                #print(task_param)
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.config['reg_coef'] * reg_loss
        
        return loss
    

'''
Implementation of Elastic weight consolidation
'''
class EWC(L2):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, agent_config):
        super(EWC, self).__init__(agent_config)
        self.online_reg = False
        self.n_fisher_sample = None
        self.empFI = False
        
    
    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        self.log('Computing EWC')
        
        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms) > 0 : 
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                # initialize with zeroes
                importance[n] = p.clone().detach().fill_(0)
        
        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size = 1)
        # Otherwise it uses mini-batches for estimation
        # This speeds up the process significantly with similar performance
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            self.log('Sample', self.n_fisher_sample, 'for estimating the Fisher matrix.')
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)
        
        mode = self.training
        self.eval()
        
        # Accumulate the square of gradients
        for i, (inputs, target) in enumerate(dataloader):
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()
                
            preds = self.forward(inputs)
            
            # masking inactive nodes
            pred = preds[:,self.active_out_nodes]
            # getting the highest prediction for each output
            ind = pred.max(1)[1].flatten()
            
            # use the groundtruth label
            # default is to not use this
            if self.empFI:
                ind = target
            
            loss = self.criterion(preds, ind, regularization = False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                # some output nodes can have no grad if no loss is applied on them
                if self.params[n].grad is not None:
                    p += ((self.params[n].grad ** 2) * len(inputs) / len(dataloader))
                    
        self.train(mode = mode)
        
        return importance
        

def EWC_online(agent_config):
    agent = EWC(agent_config)
    agent.online_reg = True
    return agent


class SI(L2):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """
    
    def __init__(self, agent_config):
        super(SI, self).__init__(agent_config)
        # original SI works in an online updating fashion
        self.online_reg = True
        self.damping_factor = 0.1
        # to store the importance of parameters (little omega in SI paper)
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()
        
        # the initial params will only be used in the first task (when the regularization terms are empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()
            
    
    # overridding the update model function
    def update_model(self, out, targets):
        
        unreg_gradients = {}
        
        # 1. Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()
            
        # 2. Collect the gradients without regularization term
        loss = self.criterion(out, targets, regularization = False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph = True)
        for n, p in self.params.items():
            # some parameters don't have gradients attributed to them
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, regularization = True)   
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            # in case some heads accumulate no gradients
            if n in unreg_gradients.keys():
                self.w[n] -= unreg_gradients[n] * delta
        
        return loss.detach()
                
                
    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg, 'SI needs online_reg = True'
        
        # initialize the importance matrix
        
        # the case of after the first task
        if len(self.regularization_terms) > 0:
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        # the current task is the first task
        else:
            importance = {}
            for n, p in self.params.items():
                # initialize importance with zeroes
                importance[n] = p.clone().detach().fill_(0)
            prev_params = self.initial_params
        
        # calculate / accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
            self.w[n].zero_()
            
        return importance
    
    
class MAS(L2):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """
    
    def __init__(self, agent_config):
        super(MAS, self).__init__(agent_config)
        self.online_reg = True
        
    
    def calculate_importance(self, dataloader):
        self.log('Computing MAS')
        
        # initialize the importance matrix
        if self.online_reg and len(self.regularization_terms) > 0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n,p in self.params.items():
                # init with zeroes
                importance[n] = p.clone().detach().fill_(0)
        
        mode = self.training
        self.eval()
        
        # accumulate the gradients of the L2 loss on the outputs
        for i, (input, target) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()
                
            pred = self.forward(input)
            
            # mask inactive nodes
            pred = pred[:,self.active_out_nodes]
            
            # compute L2 of outputs
            pred.pow_(2)
            loss = pred.mean()
            
            self.model.zero_grad()
            loss.backward()
            
            for n, p in importance.items():
                # some nodes can have no grad if inactive
                if self.params[n].grad is not None:
                    p += (self.params[n].grad.abs() / len(dataloader))
                    
                    
                    
                    
                    
                    
        self.train(mode = mode)

        return importance        
        
            
            
