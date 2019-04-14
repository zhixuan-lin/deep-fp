import torch
from torch import nn


class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, result_classes_count, num_layer):
        super(LstmNet, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, result_classes_count)
        # maps = [1, 32, 64, 128, 256]
        maps = [1, 16, 32, 64]

        self.block1 = self.__make_block(maps[0], maps[1], "elu")
        self.block2 = self.__make_block(maps[1], maps[2], "relu")
        self.block3 = self.__make_block(maps[2], maps[3], "relu")
        # self.block4 = self.__make_block(maps[3], maps[4], "relu")

    def forward(self, x):
        """
        :param x: (B, 1, N)
        :return:
        """
        B, _, N = x.size()
        N = 4096
        # cut data
        split = N // self.input_size
        x = x[:, :, :split * self.input_size]
        # conv first (B, 4, N / 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        x = x.view(B, split, self.input_size)
        lstm_output, (h_n, c_n) = self.lstm(x)
        result = self.out(lstm_output[:, -1, :])
        return result
    
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



if __name__ == "__main__":
    batch_size = 32
    input_size = 5000
    sequence_size = 4
    hidden_size = 25

    net = LstmNet(input_size=input_size, hidden_size=hidden_size, result_classes_count=100)
    x = torch.rand(batch_size, sequence_size, input_size)
    hidden = (torch.rand(1, batch_size, hidden_size), torch.rand(1, batch_size, hidden_size))
    output = net(x, hidden)
    print(output.size())
