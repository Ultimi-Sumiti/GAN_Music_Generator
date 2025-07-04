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
            
            # nn.Unflatten(1, (1, 256, 2)) 
           
            # Compressed layer
            nn.Linear(in_features=512, out_features=2),
            nn.ReLU(), 
            # Reshape torch.rand(N, 2) --> torch.rand(N, 1 , 1 , 2)
            nn.Unflatten(1,(1 , 1 , 2))

        )

        self.transp_conv_net = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(128,1), stride=1),
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
            print(y.shape)
            return y
        
# Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(128,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=14, out_channels=77, kernel_size=(1,4), stride=2),
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

  

# Merging all togheter to expolit, building the entire GAN architechture with the lightining module
# As function of this class we can directly implement the training process
class GAN(L.LightningModule):
    # Constructor
    def __init__(
        self,
        # Dim of the noise vector used as input in the generator
        latent_dim: int = 100,
        # Image output shape: 
        #channels,
        #width,
        #height,

        # Learning rate (to tune) 
        lr: float = 0.0002,
        # Adam optimizer params (to tune)
        b1: float = 0.5,
        b2: float = 0.999,

        # Minibatch size
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        # Heridarety: to initialize correctly the superclass 
        super().__init__()

        # Function to save all the hyperpar passed in the above constructor
        self.save_hyperparameters()
        
        # To control manually the optimization step, since in GAN we have to use criteria
        # in optimizing both Generator and discriminator
        self.automatic_optimization = False

        # Newtorks definition(Generator + Discriminator)
        # data_shape = (channels, width, height)
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

        #
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    # Forward step computed     
    def forward(self):
        return

    # Loss of the Adversarial game 
    def adversarial_loss(self,):
        return

    # Gan training algorithm    
    def training_step(self, ) :
        return 

    def validation_step(self, ) :
        return 

    def configure_optimizers(self) :
        return 
    
    def on_validation_epoch_end(self) :
        return 
        

# Test Main
if __name__== "__main__":

    net_generator = Generator(100)
    net_discriminator = Discriminator()
    
    x = torch.rand(1, 100)
    out = net_generator(x)

    print("Generator output ", out.size())

