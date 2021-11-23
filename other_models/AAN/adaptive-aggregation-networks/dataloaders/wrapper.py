import torch
import torch.utils.data as data

class Storage(data.Subset):

    def reduce(self, m):
        self.indices = self.indices[:m]