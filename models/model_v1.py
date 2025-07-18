import os
import sys

import numpy as np 
import torch 
import torch.nn as nn
#import lightning as L
import pytorch_lightning as L

# Used to print during training generated images.
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from IPython import display

# Import the utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.architectural_utils import *

# model_v1 used to do melodies generation

#GENERATOR ARCHITECTURE
class Generator(nn.Module):
    def __init__(self, input_size): 
        super().__init__()

        # Channels of the X in input, in our case it's = 1
        self.input_size = input_size

        self.fc_net = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128 , 1 , 2))
        )

        self.transp_conv_net = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(128,1), stride=1),
            nn.LeakyReLU(), 
            #nn.Sigmoid()
        )

        self.monophonic = MonophonicLayer()

    # Override of the forward method
    def forward(self, x):
        y = self.fc_net(x)
        y = self.transp_conv_net(y)
        y = self.monophonic(y)
        assert (y <= 1).all(), "Found a value bigger than one"
        return y

 
# Discriminator Architecture - No minibatch disc.
#class Discriminator(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        self.conv_net1 = nn.Sequential(
#            # Default: padding=0,  dilation=1
#            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(128,2), stride=2),
#            nn.LeakyReLU(), 
#            nn.Dropout(0.3),
#        )
#        
#        self.conv_net2 = nn.Sequential(
#            nn.Conv2d(in_channels=14, out_channels=77, kernel_size=(1,4), stride=2),
#            nn.LeakyReLU(), 
#            nn.Dropout(0.3),
#
#            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231) 
#            nn.Flatten(start_dim=1)
#        )
#
#        self.fc_net = nn.Sequential(
#            nn.Linear(in_features=231, out_features=1024),
#            nn.LeakyReLU(), 
#            nn.Dropout(0.3),
#            nn.Linear(in_features=1024, out_features=1),
#            #nn.Sigmoid()
#        )
#    
#    # Override of the forward method
#    def forward(self, x, feature_out=False):
#        y = self.conv_net1(x)
#
#        if feature_out:
#            return y
#        
#        y = self.conv_net2(y)
#        y = self.fc_net(y)
#        return y


# Discriminator Architecture - Minibatch disc.
class Discriminator(nn.Module):
    def __init__(self, minibatch_B=10, minibatch_C=5):
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

        minibatch_A = 231
        self.bd_net = MinibatchDiscrimination(minibatch_A, minibatch_B, minibatch_C)
        

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=231 + minibatch_B, out_features=1024),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=1),
            #nn.Sigmoid()
        )

    
    # Override of the forward method
    def forward(self, x, feature_out=False):
        y = self.conv_net1(x)

        if feature_out:
            return y
        
        features_flattened = self.conv_net2(y) 

        mbd_output = self.bd_net(features_flattened) 

        combined_features = torch.cat((features_flattened, mbd_output), dim=1) 
        
        output = self.fc_net(combined_features)
        
        return output


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
        lr_d: float = 0.0002,
        lr_g: float = 0.0002,
        # Adam optimizer params (to tune)
        b1: float = 0.5,
        b2: float = 0.999,

        # Number of updates.
        gen_updates: int = 1,
        dis_updates: int = 1,

        # Feature loss.
        lambda_1: float = 1,
        lambda_2: float = 0.1,

        # Minibatch discrimination.
        minibatch_B: int = 10,
        minibatch_C: int = 5,

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
        self.discriminator = Discriminator(
            minibatch_B, minibatch_C
        )

        # Used to see intermediate outputs after each epoch.
        self.validation_z = torch.randn(10, latent_dim)

    # Forward step computed     
    # TO check: __call__ = forward
    def forward(self, x):
        return self.generator(x)

    # Loss of the Adversarial game.
    def adversarial_loss(self, y_hat, y):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(y_hat, y)

    def validation_step(self, ) :
        pass

    def configure_optimizers(self) :
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    # It shuld be on_validation_epoch_end
    def on_train_epoch_end(self) :
        # Clear ouput.
        display.clear_output(wait=True)
        
        z = self.validation_z.type_as(self.generator.fc_net[0].weight)

        # Generate images.
        sample_curr = self(z).detach().cpu()
        
        # Grid dimensions.
        cols = 5
        rows = 2

        # Create the figure.
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2))
        axes = axes.flatten()
        for idx, (ax, img) in enumerate(zip(axes, sample_curr)):
            img_np = img.squeeze().numpy()
            im = ax.imshow(img_np, aspect='auto', origin='lower', cmap='hot')
            ax.set_title(f"#{idx}")
            fig.colorbar(im, ax=ax, label='Velocity')

        # Plot the figure.
        plt.tight_layout()
        plt.show()

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images (for the discriminator).
        curr = batch # Ignore the labels.
        
        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Measure discriminator's ability to classify real from generated samples.
    
            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(curr)
        
            # Generate new images.
            self.generated_curr = self(z)
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_d)
        
            valid = torch.ones(curr.size(0), 1) * 0.9
            valid = valid.type_as(curr)
        
            # First term of discriminator loss.
            real_loss = self.adversarial_loss(
                    self.discriminator(curr),
                    valid
            )
        
            fake = torch.zeros(curr.size(0), 1)
            fake = fake.type_as(curr)
        
            # Second term of discriminator loss.
            fake_loss = self.adversarial_loss(
                    self.discriminator(self.generated_curr.detach()),
                    fake
            )
        
            # Total discriminator loss.
            d_loss = real_loss + fake_loss

            # WGAN loss.
            #d_loss = torch.mean(self.discriminator(self.generated_curr.detach())) - torch.mean(self.discriminator(curr))
        
            # Discriminator training.
            self.log("d_loss", d_loss, prog_bar=True)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            ## WGAN.
            #for p in self.discriminator.parameters():
            #    p.data.clamp_(-0.01, 0.01)


        ### GENERATOR ####
        for _ in range(self.hparams.gen_updates):
            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(curr)
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_g)
            # Generate images.
            self.generated_curr = self(z)
            
            # Log sampled images.
            # TODO
            
            # ground truth result (ie: all fake)
            valid = torch.ones(curr.size(0), 1)
            # put on GPU because we created this tensor inside training_loop
            valid = valid.type_as(curr)
        
            # Compute the loss function.
            mean_img_from_batch = torch.mean(curr)
            mean_img_from_g = torch.mean(self.generated_curr)
            regularizer_1 = torch.norm(mean_img_from_batch - mean_img_from_g) ** 2
    
            # TOFIX: should we detach??
            real_features = torch.mean(self.discriminator(curr, feature_out=True))
            fake_features = torch.mean(self.discriminator(self.generated_curr, feature_out=True))
            regularizer_2 = torch.norm(real_features - fake_features) ** 2
            
            g_adversarial_loss = self.adversarial_loss(
                self.discriminator(self.generated_curr),
                valid
            )
    
            ## Total loss.
            g_loss = (self.hparams.lambda_1 * regularizer_1 
                     + self.hparams.lambda_2 * regularizer_2 
                     + g_adversarial_loss)
    
            # WGAN.
            #g_loss = -torch.mean(self.discriminator(self.generated_curr))
            
            # Generator training.
            self.log("g_loss", g_loss, prog_bar=True) # Log loss.
            self.manual_backward(g_loss) # Toggle.
            optimizer_g.step() # Update weights.
            optimizer_g.zero_grad() # Avoid accumulation of gradients.
            self.untoggle_optimizer(optimizer_g)



