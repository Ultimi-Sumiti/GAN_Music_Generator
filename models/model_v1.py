import os
import numpy as np 
import torch 
import torch.nn as nn
import lightning as L 


# Remember a tensor with pytorch is composed as: (N x C x H x W) if is a 3d tensor

# Generator  Network
class Generator(nn.Module):
    def __init__(self, input_size): 
        super().__init__()

        # Channels of the X in input, in our case it's = 1
        self.input_size = input_size

        self.fc_net = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= 1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            
            # Reshape torch.rand(N, 512) --> torch.rand(N, 1, 2 , 256)
            nn.Unflatten(1, (1, 2, 256))
        )

        self.transp_conv_net = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose1d(in_channels=1, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=1, kernel_size=(128,1), stride=1),
            nn.ReLU()
        )
        # Function f() to create a monophonic layer by prev. feature map, i.e
        # turn off per time step all but the note with the highest activation

        # Override of the forward method
    def forward(self, x):
            
            print(x.shape)
            y = self.fc_net(x)
            print(y.shape)
            y = self.transp_conv_net(y)
            return y
        
# Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv1d(in_channels=1, out_channels=14, kernel_size=(128,2), stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=14, out_channels=77, kernel_size=(1,4), stride=2),
            nn.ReLU(),

            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231) 
            nn.Flatten(start_dim=1)
        )

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=231, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )
    
    # Override of the forward method
    def forward(self,x):

        y = self.conv_net(x)
        y = self.fc_net(y)
        return y


# Test Main
if __name__== "__main__":

    net_generator = Generator(100)
    net_discriminator = Discriminator()
    
    x = torch.rand(1, 100)
    out = net_generator(x)

    print(out.size())

