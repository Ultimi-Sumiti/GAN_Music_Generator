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
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

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
            nn.BatchNorm1d(512),
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


class MinibatchDiscrimination(nn.Module):
    def __init__(self, A, B, C):
        super(MinibatchDiscrimination, self).__init__()
        self.feat_num = A
        self.mat_cols = B
        self.mat_rows = C
        self.T = Parameter(torch.Tensor(A,B,C))
        self.reset_parameters()

    def forward(self, x):
        N = x.size(0)
        # Step 1: Reshape T for matmul: (A, B*C)
        T_reshaped = self.T.view(self.feat_num, self.mat_cols * self.mat_rows)

        # Step 2: Perform matrix multiplication: x (N, A) @ T_reshaped (A, B*C) -> (N, B*C)
        matmul_output = x.matmul(T_reshaped)

        # Step 3: Reshape the matmul output to (N, B, C)
        M = matmul_output.view(N, self.mat_cols, self.mat_rows)

        # Computing the difference between the tensor product (in the score function)
        M_diff = M.unsqueeze(1) - M.unsqueeze(0)

        # Computing the L1 norm for the exponential in the score function
        M_diff_mod = torch.sum(torch.abs(M_diff), dim = 3)

        # Apply the actual exponential for computing the score:
        c = torch.exp(-M_diff_mod)

        # To compute the output we need to compute the sum along the first dimension of the matrices
        scores = torch.sum(c, dim = 1)

        # Now we need to compute the autocorrelation and substruct ot the scores 
        diag_elements = c[torch.arange(N, device=x.device), torch.arange(N, device=x.device)]

        out = scores - diag_elements

        return out 

    def reset_parameters(self):
        stddev = 1.0 / torch.sqrt(torch.tensor(self.feat_num, dtype=torch.float32))
        self.T.data.uniform_(-stddev, stddev)


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

        # Number of updates.
        gen_updates: int = 1,
        dis_updates: int = 1,

        # Feature loss.
        lambda_1: float = 1,
        lambda_2: float = 0.1,

        # Minibatch discrimination.
        minibatch_B: int = 10,
        minibatch_C: int = 5,

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

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images (for the discriminator).
        imgs = batch # Ignore the labels.
        
        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Measure discriminator's ability to classify real from generated samples.
    
            # Sample noise for the generator.
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(imgs)
        
            # Generate new images.
            self.generated_imgs = self(z)
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_d)
        
            valid = torch.ones(imgs.size(0), 1) * 0.9
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

            # WGAN loss.
            #d_loss = torch.mean(self.discriminator(self.generated_imgs.detach())) - torch.mean(self.discriminator(imgs))
        
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
        
            # Compute the loss function.
            mean_img_from_batch = torch.mean(imgs)
            mean_img_from_g = torch.mean(self.generated_imgs)
            regularizer_1 = torch.norm(mean_img_from_batch - mean_img_from_g) ** 2
    
            # TOFIX: should we detach??
            real_features = torch.mean(self.discriminator(imgs, feature_out=True))
            fake_features = torch.mean(self.discriminator(self.generated_imgs, feature_out=True))
            regularizer_2 = torch.norm(real_features - fake_features) ** 2
            
            g_adversarial_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs),
                valid
            )
    
            ## Total loss.
            g_loss = (self.hparams.lambda_1 * regularizer_1 
                     + self.hparams.lambda_2 * regularizer_2 
                     + g_adversarial_loss)
    
            # WGAN.
            #g_loss = -torch.mean(self.discriminator(self.generated_imgs))
            
            # Generator training.
            self.log("g_loss", g_loss, prog_bar=True) # Log loss.
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