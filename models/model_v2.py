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

# Strategy to bypass the problem of the thrashold wich causes non differentiable function
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

# Function to concatenate the conditioning 2D of the Conditioner (Sistemare posizione di qeusta funzione) TODO
def conv_prev_concat(x, cond_x): # cond_x is the output tensor of a certain layer of the Conditioner
    
    x_shapes = x.shape
    cond_x_shapes = cond_x.shape
    if x_shapes[2:] == cond_x_shapes[2:]:
        new_cond_x = cond_x.expand(x_shapes[0],cond_x_shapes[1],x_shapes[2],x_shapes[3])
        # The size of the new tensor will be [batch_size * (C_x + C_cond_x) * Height * Weight]
        return torch.cat((x, new_cond_x),1)

    else:
        print("Problem in the match of x_shapes[2:] with new_cond_x[2:]")
        print(x_shapes[2:])
        print(cond_x_shapes[2:])


# Used to init the weights of Discriminator and Generator.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)

# Generator  Network
class Generator(nn.Module):
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
            nn.BatchNorm2d(a), # TO CHECK TODO
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


# IMPORTANTE CAMBIATO in_channels, perche dvee prendere la somma della conv concat  ora TODO        
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
    # x is the noise, prev_x is the previously generated sample obtained from ... TODO
    def forward(self, x):
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
        
        y = conv_prev_concat(y, cond4)                                              # ([bs, a + w_size, 1, 2])
        #print("\ny after first conditional concat:", y.size())
        y = self.transp_conv1(y)                                                    ## ([bs, w_size, 1, 4])
        
        # Concatenate conv3 conditioner with y (:= output tensor of transp_conv1)
        #print("y after the first transp conv", y.size())
        y = conv_prev_concat(y, cond3)                                              # ([bs, a + w_size, 1, 4])
        #print("\ny after second conditional concat:", y.size()) 
        y = self.transp_conv2(y)                                                    ## ([bs, w_size, 1, 8])

        # Concatenate conv2 conditioner with y (:= output tensor of transp_conv2)
        #print("y after the second transp conv", y.size())
        y = conv_prev_concat(y, cond2)                                              # ([bs, a + w_size, 1, 8])
        #print("\ny after third conditional concat:", y.size()) 
        y = self.transp_conv3(y)                                                    ## ([bs, w_size, 1, 16])

        # Concatenate conv1 conditioner with y (:= output tensor of transp_conv3)
        #print("y after the third transp conv", y.size())
        y = conv_prev_concat(y, cond1)                                              # ([bs, a + w_size, 1, 16])
        #print("\ny after fourth conditional concat:", y.size()) 
        y = self.transp_conv4(y)                                                    ## ([bs, 1, 128, 16])
        
        #print("y after last transp conv (4)", y.size())

        y = self.monophonic(y)                                                      # ([bs, 1, 128, 16])
        #print("\ny after monophonic", y.size())
        #assert (y <= 1).all(), "Found a value bigger than one"
        return y


# Discriminator Architecture - Minibatch disc.
class Discriminator(nn.Module):
    def __init__(self, apply_mbd=False, mbd_B_dim=10, mbd_C_dim=5):
        super().__init__()

        # True if one want to use minibatch discrimination.
        self.apply_mbd = apply_mbd

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
        
        # Noise vector dim for generator.
        latent_dim: int = 100,
        
        # Learning rate.
        lr: float = 0.0002,
        
        # Adam optimizer params.
        b1: float = 0.5,
        b2: float = 0.999,

        # Feature loss params.
        lambda_1 = 0.1,
        lambda_2 = 1,

        # Number of updates per iteration.
        gen_updates: int = 1,
        dis_updates: int = 1,

        apply_mbd: bool = False,
        mbd_B_dim: int = 10,
        mbd_C_dim: int = 5,

        a: int = 16,
        
        # Minibatch size.
        batch_size: int = BATCH_SIZE,
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

        # Used to see intermediate outputs after each epoch.
        #self.validation_z = torch.randn(10, latent_dim)

    # Forward step computed     
    def forward(self, x):
        return self.generator(x)

    # Loss of the Adversarial game.
    def adversarial_loss(self, y_hat, y):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(y_hat, y)

    # Gan training algorithm    
    def training_step(self, batch):
        # Batch images.
        prev, imgs = batch

        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Sample noise for the generator.
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(imgs)
        
            # Generate new images.
            self.generated_imgs = self.generator((z, prev))
            
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
        
            # Discriminator training.
            self.log("d_loss", d_loss, prog_bar=True)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        ### GENERATOR ####
        for _ in range(self.hparams.gen_updates):
            # Sample noise for the generator.
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
            # put on GPU because we created this tensor inside training_loop
            z = z.type_as(imgs)
            
            # Activate Generator optimizer. 
            self.toggle_optimizer(optimizer_g)
            # Generate images.
            self.generated_imgs = self.generator((z, prev))
            
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
        
            # Total loss.
            g_loss = (self.hparams.lambda_1 * regularizer_1 
                      + self.hparams.lambda_2 * regularizer_2 
                      + g_adversarial_loss)
            
            # Generator training.
            self.log("g_loss", g_loss, prog_bar=True) # Log loss.
            self.manual_backward(g_loss) # Toggle.
            optimizer_g.step() # Update weights.
            optimizer_g.zero_grad() # Avoid accumulation of gradients.
            self.untoggle_optimizer(optimizer_g)


    def validation_step(self, ) :
        pass

    def configure_optimizers(self) :
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    ## It shuld be on_validation_epoch_end
    #def on_train_epoch_end(self) :
    #    # Clear ouput.
    #    display.clear_output(wait=True)
    #    
    #    z = self.validation_z.type_as(self.generator.fc_net[0].weight)
#
    #    # Generate images.
    #    sample_imgs = self(z).detach().cpu()
#
    #    # Grid dimensions.
    #    cols = 5
    #    rows = 2
#
    #    # Create the figure.
    #    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2))
    #    axes = axes.flatten()
    #    for idx, (ax, img) in enumerate(zip(axes, sample_imgs)):
    #        img_np = img.squeeze().numpy()
    #        im = ax.imshow(img_np, aspect='auto', origin='lower', cmap='hot')
    #        ax.set_title(f"#{idx}")
    #        fig.colorbar(im, ax=ax, label='Velocity')
#
    #    # Plot the figure.
    #    plt.tight_layout()
    #    plt.show()



# Tester to see if the dimensions are correct
if __name__ == "__main__": 
    # Note we have to reason considering the w_size as fixed to 128 to modify the other params of the net
    # Generator(#dim of noise, #w_size of the image, in our case is always 128, #num of kernels of the conditioner layers)
    model = Generator(100,128,25)
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


        

        



        