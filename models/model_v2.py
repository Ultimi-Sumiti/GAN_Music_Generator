""" This file contains all the code for the creation of a GAN to produce melodies
of 8 bars (16 time steps and 128 notes/semitones).

Here we train the GAN (generator and discriminator) with a dataset of samples made of pairs of bars, obtained by processing midi files in the original dataset and converting the pretty_midi objects into 1x128x16 tensors. 
Those pairs of bars which becomes pairs of tensors are couples of subsequent bars in one of the input midi files.

The main difference between this model and the previous is that here we need to consider the relations between current and previous bar in the midi dataset, to achieve this we used a conditional GAN embedded in the generator.

Another important thing to consider when reading the code below is that some of the typical
steps done when coding a NN are skipped because they are implicitely done by the pytorch_lightning
library.
"""

# System libraries.
import os
import sys

# Data/staticts/NN libraries.
import numpy as np 
# Remember a tensor with pytorch is composed as: (N x C x H x W) if is a 3d tensor
import torch 
import torch.nn as nn
#import lightning as L
import pytorch_lightning as L

# Import the utils.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.architectural_utils import *




# GENERATOR ARCHITECTHURE
class Generator(nn.Module):
    """This class represent the GAN generator, it's an extension of the torch.nn module.
    It is composed by three groups of layers:
        -a group of linear layers
        -a group of transpose convolutional layers
        -a custom monophonic layer
    Both activations (LeakyRelu) and batch normalization are used.

    Attributes:
        input_size        The size of the input noise for the generator.
        fc_net            The group of linear layers.
        transp_conv_net   The group of transpose convolution layers.
        monophonic        The final custom Function to avoid vanishing gradients and to
                          get only one note per time step.
    """
    
    # a := num Channels of the conditioner cnn layers (num of kernels)
    # w_size := dim of the w size of the image, in our case is fixed always to 128     
    # input_size := dim of the noise vector takes in input
    def __init__(self, input_size, w_size=128, a=16): 
        super().__init__()

        # Channels of the X in input, in our case it's = 1
        self.input_size = input_size
        self.pitch_size = w_size
        self.a = a
       
        self.transp_layer_size = self.a + w_size

        # Conditioner CNN
        self.conv1_cond = nn.Sequential(
            nn.Conv2d(in_channels=1 , out_channels=a, kernel_size=(128,1), stride=1),
            nn.BatchNorm2d(a),
            nn.LeakyReLU() 

        ) 
        self.conv2_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1,2), stride=2),  
            nn.BatchNorm2d(a),
            nn.LeakyReLU() 
        ) 

        self.conv3_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(a),
            nn.LeakyReLU() 
        ) 
        self.conv4_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(a),
            nn.LeakyReLU() 
        )
        
        # MLP 
        self.fc_net = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            
            nn.Unflatten(1, (128, 1, 2)) 
            
        )

        # Transpse convolution layers
        self.transp_conv1 = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(in_channels=self.transp_layer_size, out_channels=w_size, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU()
        ) 
        self.transp_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.transp_layer_size, out_channels=w_size, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU()
        ) 

        self.transp_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.transp_layer_size, out_channels=w_size, kernel_size=(1,2), stride=2),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU() 
        )  
            
        self.transp_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.transp_layer_size, out_channels=1, kernel_size=(w_size,1), stride=1),
            #nn.BatchNorm2d(1), # TO CHECK TODO
            nn.LeakyReLU()
        )
            
        self.monophonic = MonophonicLayer()

    # Override of the forward method
    def forward(self, x):
        # x is the noise, prev_x is the previously generated sample obtained from the pair of bars
        # (prev, curr), in this case we feed the conditioner with the prev, while the discriminator with curr 
        prev_x = x[1]
        x = x[0]

        # Process the previous generated sample in the Conditioner network
        # First number is the expected according to the paper, second number the one according to be consistent with the implementation
        cond1 = self.conv1_cond(prev_x)                                              # ([bs, a, 1, 16])
        #print("\nDime cond1", cond1.size())
        cond2 = self.conv2_cond(cond1)                                               # ([bs, a, 1, 8])
        #print("Dim cond2", cond2.size())
        cond3 = self.conv3_cond(cond2)                                               # ([bs, a, 1, 4])
        #print("Dim cond3", cond3.size())
        cond4 = self.conv4_cond(cond3)                                               #  ([bs, a, 1, 2])
        #print("Dime cond4", cond4.size())

        # At the end we must have that a + b = c , where b is the dim of channel transp. conv layer, and c is the sum
        # between conditioner layer channel (a) and (b)

        y = self.fc_net(x)                                                          # ([bs, w_size, 1, 2])

        # Concatenate conv4 conditioner with y (:= output tensor of fc_net) 
        #print("\ny after the MLP", y.size())
        
        y = prev_concat(y, cond4)                                              # ([bs, a + w_size, 1, 2])
        #print("\ny after first conditional concat:", y.size())
        y = self.transp_conv1(y)                                                    ## ([bs, w_size, 1, 4])
        
        # Concatenate conv3 conditioner with y (:= output tensor of transp_conv1)
        #print("y after the first transp conv", y.size())
        y = prev_concat(y, cond3)                                              # ([bs, a + w_size, 1, 4])
        #print("\ny after second conditional concat:", y.size()) 
        y = self.transp_conv2(y)                                                    ## ([bs, w_size, 1, 8])

        # Concatenate conv2 conditioner with y (:= output tensor of transp_conv2)
        #print("y after the second transp conv", y.size())
        y = prev_concat(y, cond2)                                              # ([bs, a + w_size, 1, 8])
        #print("\ny after third conditional concat:", y.size()) 
        y = self.transp_conv3(y)                                                    ## ([bs, w_size, 1, 16])

        # Concatenate conv1 conditioner with y (:= output tensor of transp_conv3)
        #print("y after the third transp conv", y.size())
        y = prev_concat(y, cond1)                                              # ([bs, a + w_size, 1, 16])
        #print("\ny after fourth conditional concat:", y.size()) 
        y = self.transp_conv4(y)                                                    ## ([bs, 1, 128, 16])
        
        #print("y after last transp conv (4)", y.size())

        y = self.monophonic(y)                                                      # ([bs, 1, 128, 16])
        #print("\ny after monophonic", y.size())
        #assert (y <= 1).all(), "Found a value bigger than one"
        return y


# DISCRIMINATOR ARCHITECHTURE
class Discriminator(nn.Module):
    def __init__(self, apply_mbd=False, mbd_B_dim=10, mbd_C_dim=5):
        super().__init__()

        # True if one want to use minibatch discrimination.
        self.apply_mbd = apply_mbd

        self.conv_net1 = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(128,2), stride=2),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
        )
        
        self.conv_net2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=77, kernel_size=(1,3), stride=2),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),

            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231) 
            nn.Flatten(start_dim=1)
        )

        # If minibatch discrimination is applied.
        if apply_mbd:
            mbd_A_dim = 231
            self.bd_net = MinibatchDiscrimination(mbd_A_dim, mbd_B_dim, mbd_C_dim)

            self.fc_net = nn.Sequential(
                nn.Linear(in_features=231 + mbd_B_dim, out_features=1024),
                nn.LeakyReLU(), 
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=1),
                #nn.Sigmoid()
            )
            
        # No minibatch discrimination is applied.
        else:
            self.fc_net = nn.Sequential(
                nn.Linear(in_features=231, out_features=1024),
                nn.LeakyReLU(), 
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=1),
                #nn.Sigmoid()
            )
    
    # Override of the forward method
    def forward(self, x, feature_out=False):
        y = self.conv_net1(x)

        # Needed for the feature loss in GAN training.
        if feature_out:
            return y

        # If we apply minibatch discrimination.
        if self.apply_mbd:
            features_flattened = self.conv_net2(y) 
    
            mbd_output = self.bd_net(features_flattened) 
    
            combined_features = torch.cat((features_flattened, mbd_output), dim=1) 
            
            y = self.fc_net(combined_features)

        # No minibatch discrimination is applied.
        else:
            y = self.conv_net2(y)
            y = self.fc_net(y)
        
        return y


# Merging all togheter to expolit, building the entire GAN architechture with the lightining module
# As function of this class we can directly implement the training process
class GAN(L.LightningModule):

    # Constructor
    def __init__(
        self,
        
        # Noise vector dim for generator.
        latent_dim: int = 100,
        
        # Learning rate.
        lr_d: float = 0.0002,
        lr_g: float = 0.0002,
        
        # Adam optimizer params.
        b1: float = 0.5,
        b2: float = 0.999,

        # Feature loss params.
        lambda_1 = 0.1,
        lambda_2 = 1,

        # Number of updates per iteration.
        gen_updates: int = 1,
        dis_updates: int = 1,

        # Minibatch discrimination.
        apply_mbd: bool = False,
        mbd_B_dim: int = 10,
        mbd_C_dim: int = 5,

        a: int = 16,

        **kwargs,
    ):
        # Heridarety: to initialize correctly the superclass 
        super().__init__()

        # Function to save all the hyperpars passed in constructor.
        self.save_hyperparameters()
        
        # To control manually the optimization step, since in GAN we have to use criteria
        # in optimizing both Generator and discriminator
        self.automatic_optimization = False

        # Newtorks definition(Generator + Discriminator)
        # data_shape = (channels, width, height)
        self.generator = Generator(latent_dim, a=a)
        self.discriminator = Discriminator(
            apply_mbd=apply_mbd,
            mbd_B_dim=mbd_B_dim,
            mbd_C_dim=mbd_C_dim
        )

    
    # Forward step computed     
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

        scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=4000, gamma=0.7)
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=4000, gamma=0.7)
        return ([opt_g, opt_d], [scheduler_g, scheduler_d])
        

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images.
        prev, curr = batch

        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(curr)
        
            # Generate new images.
            self.generated_curr = self.generator((z, prev))
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_d)
            
            # One-sided label smoothing
            valid = torch.ones(curr.size(0), 1) * 0.9
            valid = valid.type_as(curr)

            # maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator(curr)
            fake_pred = self.discriminator(self.generated_curr.detach())
            
            # First term of discriminator loss.
            real_loss = self.adversarial_loss(real_pred, valid)
        
            fake = torch.zeros(curr.size(0), 1)
            fake = fake.type_as(curr)
            
            # Second term of discriminator loss.
            fake_loss = self.adversarial_loss(fake_pred, fake)
            
            # Total Discriminator loss.
            d_loss = real_loss + fake_loss

            # Discriminator training iteration step.
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            # Compute confidence.
            conf_real = torch.sigmoid(real_pred).mean()
            conf_fake = 1 - torch.sigmoid(fake_pred).mean()

            # Log info.
            self.log("d_loss", d_loss, prog_bar=True)
            self.log("conf_real", conf_real, prog_bar=True)
            self.log("conf_fake", conf_fake, prog_bar=True)

        ### GENERATOR ####
        for _ in range(self.hparams.gen_updates):
            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(curr)
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_g)
            
            # Generate images. # this could be also non self (modify) 
            self.generated_curr = self.generator((z, prev))
            
            # ground truth result (ie: all fake)
            valid = torch.ones(curr.size(0), 1)
            # put on GPU because we created this tensor inside training_loop
            valid = valid.type_as(curr)

            # maximize log(D(G(z))) + lamda1 * ||mean_img_from_batch - mean_img_from_g||_2 + lamda2 * ||real_features - fake_features||_2
            # Compute the loss function with the application of Feature Matching techinque
            mean_img_from_batch = torch.mean(curr, 0)
            mean_img_from_g = torch.mean(self.generated_curr, 0)
            regularizer_1 = torch.norm(mean_img_from_batch - mean_img_from_g) ** 2
        
            real_features = torch.mean(self.discriminator(curr, feature_out=True), 0)
            fake_features = torch.mean(self.discriminator(self.generated_curr, feature_out=True), 0)
            regularizer_2 = torch.norm(real_features - fake_features) ** 2
            
            g_adversarial_loss = self.adversarial_loss(
                self.discriminator(self.generated_curr),
                valid
            )
        
            # Total Generator loss.
            g_loss = (self.hparams.lambda_1 * regularizer_1 
                      + self.hparams.lambda_2 * regularizer_2 
                      + g_adversarial_loss)

            # Generator training iteration step. 
            self.log("g_loss", g_loss, prog_bar=True) # Log loss.
            self.manual_backward(g_loss) # Toggle.


            # Compute gradient norm.
            grad_norm = get_gradient_norm(self.generator)
            self.log("grad_norm", grad_norm, prog_bar=True)
            
            optimizer_g.step() # Update weights.
            optimizer_g.zero_grad() # Avoid accumulation of gradients.
            self.untoggle_optimizer(optimizer_g)




# Tester to see if the dimensions are correct
if __name__ == "__main__": 
    # Note we have to reason considering the w_size as fixed to 128 to modify the other params of the net
    # Generator(#dim of noise, #w_size of the image, in our case is always 128, #num of kernels of the conditioner layers)
    model = Generator(100, 128, 25)
    model.eval() # Set to evaluation mode to test, othewise BatchNorm can't work

    # Note, is IMPORTANT that conv_x and z have the same BATCH_SIZE
    z = torch.randn(5,100)
    #print(z)
    #print(z.size())

    conv_x = torch.zeros(5,1,128,16)
    #print(conv_x)

    # For how is implemented the dataset 2 we need to pass noise + prev as a pair
    pair = (z,conv_x)

    # Call to forward of the generator
    y = model(pair)
    #print(y)

    model = GAN()
    
    y = model(pair)
    print(y)
