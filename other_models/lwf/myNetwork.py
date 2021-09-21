import torch.nn as nn

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        # print("in network print numclass")
        # print(numclass)
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        # print("n forward print input size")
        # print(input.size())
        x = self.feature(input)
        # print("n forward print 1st x size")
        # print(x.size())
        x = self.fc(x)
        # print("in forward print 2nd x size")
        # print(x.size())
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        # print("in incremental learning")
        # print("numclasses")
        # print(numclass)
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)
