import os
import numpy as np 
import torch 
import torch.nn as nn
#import lightning as L
import pytorch_lightning as L

# Used to print during training generated images.
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from IPython import display
from torch.autograd import Function

# Define batch size.
BATCH_SIZE = 32

class MonophonicSTE(Function):
    @staticmethod
    def forward(ctx, x):
        # x: (N, C, F, T) for example
        # Find the max along the feature axis (dim=2)
        _, max_idx = x.max(dim=2, keepdim=True)
        # Create a hard one‐hot tensor
        y_hard = torch.zeros_like(x)
        y_hard.scatter_(2, max_idx, 1.0)
        # Save nothing for backward, since gradient is identity
        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        # Straight‐through: pass the gradient through unchanged
        return grad_output

class MonophonicLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Apply the Function; during forward you get the hard mask,
        # but in backward the grad w.r.t. x is grad_output (identity).
        return MonophonicSTE.apply(x)

# Remember a tensor with pytorch is composed as: (N x C x H x W) if is a 3d tensor

# Generator  Network
class Generator(nn.Module):
    def __init__(self, input_size): 
        super().__init__()

        # Channels of the X in input, in our case it's = 1
        self.input_size = input_size

        self.fc_net = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            
            # nn.Unflatten(1, (1, 256, 2)) 
           
            # Compressed layer
            nn.Linear(in_features=512, out_features=2),
            nn.LeakyReLU(), 
            # Reshape torch.rand(N, 2) --> torch.rand(N, 1 , 1 , 2)
            nn.Unflatten(1,(1 , 1 , 2))

        )

        self.transp_conv_net = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(128,1), stride=1),
            nn.LeakyReLU(), 
        )

        self.monophonic = MonophonicLayer()

    # Override of the forward method
    def forward(self, x):
        y = self.fc_net(x)
        y = self.transp_conv_net(y)
        y = self.monophonic(y)
        #assert (y <= 1).all(), "Found a value bigger than one"
        return y

 
# Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net1 = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(128,2), stride=2),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
        )
        
        self.conv_net2 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=77, kernel_size=(1,4), stride=2),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),

            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231) 
            nn.Flatten(start_dim=1)
        )

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=231, out_features=1024),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=1),
            #nn.Sigmoid()
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

        # Used to see intermediate outputs after each epoch.
        self.validation_z = torch.randn(10, latent_dim)
        self.example_input_array = torch.zeros(2, latent_dim)

    # Forward step computed     
    # TO check: __call__ = forward
    def forward(self, x):
        return self.generator(x)

    # Loss of the Adversarial game.
    def adversarial_loss(self, y_hat, y):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(y_hat, y)

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images (for the discriminator).
        imgs = batch # Ignore the labels.
        
        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### GENERATOR ####
        # Sample noise for the generator.
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        # put on GPU because we created this tensor inside training_loop
        z = z.type_as(imgs)

        # Activate Generator optimizer. 
        self.toggle_optimizer(optimizer_g)
        # Generate images.
        self.generated_imgs = self(z)

        # Log sampled images.
        # TODO
        
        # ground truth result (ie: all fake)
        valid = torch.ones(imgs.size(0), 1)
        # put on GPU because we created this tensor inside training_loop
        valid = valid.type_as(imgs)

        # Generator loss.
        g_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs),
                valid
        )

        # Generator training.
        self.log("g_loss", g_loss, prog_bar=True) # Log loss.
        self.manual_backward(g_loss) # Toggle.
        optimizer_g.step() # Update weights.
        optimizer_g.zero_grad() # Avoid accumulation of gradients.
        self.untoggle_optimizer(optimizer_g)

        ### DISCRIMINATOR ####
        # Measure discriminator's ability to classify real from generated samples.
        # Activate Generator optimizer. 
        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # First term of discriminator loss.
        real_loss = self.adversarial_loss(
                self.discriminator(imgs),
                valid
        )

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        # Second term of discriminator loss.
        fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()),
                fake
        )

        # Total discriminator loss.
        d_loss = real_loss + fake_loss

        # Discriminator training.
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)


    def validation_step(self, ) :
        pass

    def configure_optimizers(self) :
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    # It shuld be on_validation_epoch_end
    def on_train_epoch_end(self) :
        # Clear ouput.
        display.clear_output(wait=True)
        
        z = self.validation_z.type_as(self.generator.fc_net[0].weight)

        # Generate images.
        sample_imgs = self(z).detach().cpu()

        # Grid dimensions.
        cols = 5
        rows = 2

        # Create the figure.
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2))
        axes = axes.flatten()
        for idx, (ax, img) in enumerate(zip(axes, sample_imgs)):
            img_np = img.squeeze().numpy()
            im = ax.imshow(img_np, aspect='auto', origin='lower', cmap='hot')
            ax.set_title(f"#{idx}")
            fig.colorbar(im, ax=ax, label='Velocity')

        # Plot the figure.
        plt.tight_layout()
        plt.show()