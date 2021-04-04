import time
import torch

def accuracy(output, target, topk=(1,)):
    '''computes precision@k for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # prediction vectors are stacked along the batch dimension (dim zero)
        _, pred = output.topk(k = maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)
            
        if len(res) == 1:
            return res[0]
        else:
            return res
        

class AverageMeter(object):
    '''
    Computes and stores the average and current values
    '''
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0    # current (latest) value
        self.avg = 0    # running average
        self.sum = 0    # running sum
        self.count = 0  # running count (number of updates)
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count
        

class Timer(object):
    '''
    Implementation of timer to time batches, epochs, tasks, etc.
    '''
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.interval = 0
        self.time = time.time()
        
    def value(self):
        return time.time() - self.time
    
    def tic(self):
        self.time = time.time()
        
    def toc(self):
        # length of time that has passed
        self.interval = time.time() - self.time
        # recording completion time
        self.time = time.time()     
        return self.interval