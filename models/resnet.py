''' 
Implementation of resnet-18
'''

import torchvision.models as models
import torch.nn as nn

#class ResNet18(nn.Module):
#    
#    def __init__(self, model_config):
#        self.config = model_config
#        super(ResNet18, self).__init__()
#        
#        self.model = models.resnet18(pretrained = self.config['pretrained'])
#        
#        # freezing weights for feature extraction if desired
#        if self.config['freeze_feature_extract']:
#            for param in self.model.parameters():
#                param.requires_grad = False
#                
#        if self.config['n_class'] is not None:
#            print("Changing output layer to contain {} classes".format(self.config['n_class']))
#            self.model.fc = nn.Linear(512, self.config['n_class'])
#        
#                        
#    def forward(self, x):
#        out = self.model(x)
#        return(out)
        
#class ResNet18(nn.Module):
#    
#    def __init__(self, model_config):
#        self.config = model_config
#        super(ResNet18, self).__init__()
#        
#        self.tempmodel = models.vgg16(pretrained = self.config['pretrained']) 
#                
#        if self.config['n_class'] is not None:
#            print("Changing output layer to contain {} classes".format(self.config['n_class']))
#            self.model = nn.Sequential(self.tempmodel.features,
#                                   self.tempmodel.avgpool,
#                                   nn.Flatten(),
#                                   *(list(self.tempmodel.classifier)[0:-1]),
#                                      nn.Linear(4096, self.config['n_class']))
#        else:
#           self.model = nn.Sequential(self.tempmodel.features,
#                                   self.tempmodel.avgpool,
#                                   nn.Flatten(),
#                                   self.tempmodel.classifier)
#           
#       # freezing weights for feature extraction if desired
#        if self.config['freeze_feature_extract']:
#            for param in self.model.parameters():
#                param.requires_grad = False
#        
#                        
#    def forward(self, x):
#        out = self.model(x)
#        return(out)

class ResNet18(nn.Module):
    
    def __init__(self, model_config):
        self.config = model_config
        super(ResNet18, self).__init__()
        
        self.model = models.squeezenet1_0(pretrained=self.config['pretrained'])
        
        # freezing weights for feature extraction if desired
        if self.config['freeze_feature_extract']:
            for param in self.model.parameters():
                param.requires_grad = False
                
        if self.config['n_class'] is not None:
            print("Changing output layer to contain {} classes".format(self.config['n_class']))
            self.model.classifier[1] = nn.Conv2d(512, self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))
            
        self.model = nn.Sequential(self.model, nn.Flatten())
                        
    def forward(self, x):
        out = self.model(x)
        return out
