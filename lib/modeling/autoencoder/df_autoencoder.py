import torch
from torch import nn
from torch.nn.functional import mse_loss


class DFAutoEncoderSimple(nn.Module):
    def __init__(self, in_features=4096):
        nn.Module.__init__(self)
        
        self.in_features = 1
        in_features = 1
        
        # maps = [32, 32, 32, 32]
        maps = [64, 64, 16]

        self.encoder = []
        for out_features in maps:
            self.encoder.append(nn.Conv1d(in_features, out_features, kernel_size=7, stride=4, padding=3))
            in_features = out_features
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        out_features = self.in_features
        for in_features in maps:
            self.decoder.insert(0, nn.ConvTranspose1d(in_features, out_features, kernel_size=7, stride=4, padding=3, output_padding=3))
            out_features = in_features

        self.decoder = nn.Sequential(*self.decoder)


        # self.in_features = in_features
        # maps = [2048, 1024, 512]
        #
        # self.encoder = []
        # for out_features in maps:
        #     self.encoder.append(nn.Linear(in_features, out_features))
        #     in_features = out_features
        # self.encoder = nn.Sequential(*self.encoder)
        #
        # self.decoder = []
        # out_features = self.in_features
        # for in_features in maps:
        #     self.decoder.insert(0, nn.Linear(in_features, out_features))
        #     out_features = in_features
        #
        # self.decoder = nn.Sequential(*self.decoder)
        
    
    def forward(self, x, targets=None):
        """
        :param x: (B, 1, N)
        """
        B, _, N = x.size()
        # x = x.view(B, N)
        original = x
        x = self.encoder(x)
        x = self.decoder(x)
        
        loss = mse_loss(x, original)
        # we will test some data
        with open('log.txt', 'w') as f:
            f.write(str(x[0][0].cpu().detach().numpy()[300:330]))
            f.write('\n')
            f.write(str(original[0][0].cpu().detach().numpy()[300:330]))
        
        return loss
    

