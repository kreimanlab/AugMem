''' 
Implementation of multilayer perceptron model
'''

import torch
import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, model_config):
        # call init to super
        super(MLP, self).__init__()
        
        self.config = model_config
                
        # predefining these parameters
        self.in_channel = 3
        self.img_sz = 224 
        self.hidden_dim = 512
        
        # MLP takes in a vector of input
        self.in_dim = self.in_channel * self.img_sz * self.img_sz
        
        # defining the main body of network
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True)
        )
        # last layer is subject to replacement depending on the task
        self.last = nn.Linear(self.hidden_dim, self.config['n_class'])
        
    # collapses image into feature vector & passes thru input/hidden layers
    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x
    
    # takes in features and converts to logits (by passing thru last layer)
    def logits(self, x):
        x = self.last(x)
        return x
    
    # defining complete forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


# defining functions that give different instances of the MLP
def MLP100():
    return MLP(hidden_dim = 100)


def MLP1000():
    return MLP(hidden_dim = 1000)

def MLP5000():
    return MLP(hidden_dim = 5000)