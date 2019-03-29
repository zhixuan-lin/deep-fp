import torch
from torch import nn


class DFNet(nn.Module):
    def __init__(self, classes):
        nn.Module.__init__(self)
        
        maps = [1, 32, 64, 128, 256]
        
        self.block1 = self.__make_block(maps[0], maps[1], "elu")
        self.block2 = self.__make_block(maps[1], maps[2], "relu")
        self.block3 = self.__make_block(maps[2], maps[3], "relu")
        self.block4 = self.__make_block(maps[3], maps[4], "relu")
        
        self.fc_block = self.__make_fc()
        self.pred = nn.Linear(512, classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1)
        x = self.fc_block(x)
        x = self.pred(x)
        
        return x
        
    def __make_block(self, maps_in, maps, act_type):
        act = {'relu': nn.ReLU(), 'elu': nn.ELU()}[act_type]
        
        return nn.Sequential(
            nn.Conv1d(maps_in, maps, kernel_size=7, padding=3),
            nn.BatchNorm1d(maps),
            act,
            nn.Conv1d(maps, maps, kernel_size=7, padding=3),
            nn.BatchNorm1d(maps),
            act,
            nn.MaxPool1d(kernel_size=7, stride=4, padding=3),
            nn.Dropout(p=0.1)
        )
    
    def __make_fc(self, in_features=256*20):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
        
if __name__ == '__main__':
    net = DFNet(100)
    input = torch.rand(32, 1, 5000)
    output = net(input)
    print(output.size())
