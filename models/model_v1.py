import os
import numpy as np 
import torch 
import torch.nn as nn
#import lightning as L 
import pytorch_lightning as L

from dataset_loader import *

# Define batch size.
BATCH_SIZE = 32

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

        # Used to see intermediate outputs after each epoch.
        self.validation_z = torch.randn(8, latent_dim)
        self.example_input_array = torch.zeros(2, latent_dim)

    # Forward step computed     
    # TO check: __call__ = forward
    def forward(self, x):
        self.generator(x)

    # Loss of the Adversarial game.
    def adversarial_loss(self, y_hat, y):
        loss_fn = nn.BCEWithLogits()
        return loss_fn(y_hat, y)

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images (for the discriminator).
        imgs = batch # Ignore the labels.
        
        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

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
        #self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)


        # Sample noise for the generator.
        # img.shape = dim batch.
        z = torch.randn(img.shape[0], self.hparams.latent_dim)
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
        #self.log("g_loss", g_loss, prog_bar=True) # Log loss.
        self.manual_backward(g_loss) # Toggle.
        optimizer_g.step() # Update weights.
        optimizer_g.zero_grad() # Avoid accumulation of gradients.
        self.untoggle_optimizer(optimizer_g)
        

    def validation_step(self, ) :
        pass

    def configure_optimizers(self) :
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    
    def on_validation_epoch_end(self) :
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)
        

# Test Main
if __name__== "__main__":

    #net_generator = Generator(100)
    #net_discriminator = Discriminator()
    
    #x = torch.rand(1, 100)
    #out = net_generator(x)

    model = GAN()

    dm = MaestroV3DataModule("../data/preprocessed/maestro-v3.0.0/dataset1/")

    trainer = L.Trainer(
        #enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=1,
    )
    trainer.fit(model, dm)





    print("Generator output ", out.size())

