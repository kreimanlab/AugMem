''' 
Implementation of mobilenetV2
'''

import torchvision.models as models
import torch.nn as nn
    
class MobileNetV2(nn.Module):
    
    def __init__(self, model_config):
        self.config = model_config
        super(MobileNetV2, self).__init__()
        
        self.model = models.mobilenet_v2(pretrained = self.config['pretrained'])
        #replace all batchnorm2d with identity placeholder
        self.model = self.convert_layers(self.model, nn.BatchNorm2d)
        
        # freezing weights for feature extraction if desired
        if self.config['freeze_feature_extract']:
            for param in self.model.parameters():
                param.requires_grad = False
          
        #changing output number of classes
        if self.config['n_class'] is not None:
            print("Changing output layer to contain {} classes".format(self.config['n_class']))
            model.classifier[1] = nn.Linear(1280,self.config['n_class'])
                                                
    def forward(self, x):
        out = self.model(x)
        return(out)
    
    def convert_layers(model, layer_type_old):
    
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # recurse
                model._modules[name] = convert_layers(module, layer_type_old)

            if type(module) == layer_type_old:
                layer_old = module
                layer_new = torch.nn.Identity() 
                model._modules[name] = layer_new

        return model 