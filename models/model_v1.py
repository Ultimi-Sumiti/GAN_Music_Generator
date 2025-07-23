""" This file contains all the code for the creation of a GAN to produce melodies
of 1 bar (16 time steps and 128 notes/semitones).

Here we train the GAN (generator and discriminator) with a dataset of bars obtained 
by processing midi files in the original dataset and converting the pretty_midi objects
into 1x128x16 tensors.

Another important thing to consider when reading the code below is that some of the typical
steps done when coding a NN are skipped because they are implicitely done by the pytorch_lightning
library.
"""

# System libraries.
import os
import sys

# Data/staticts/NN libraries.
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L

# Used to print during training generated images.
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from IPython import display

# Import the utils.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.architectural_utils import *


# GENERATOR ARCHITECTURE
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

    def __init__(self, input_size):
        """Initialize the layers and input size from argument.
        Arguments:
            input_size    The size of the input noise for the generator.
        """
        super().__init__()

        # Channels of the X in input, in our case it's = 1
        self.input_size = input_size

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, 1, 2)),
        )

        self.transp_conv_net = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=(1, 2), stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=(128, 1), stride=1
            ),
            nn.LeakyReLU(),
            # nn.Sigmoid()
        )

        self.monophonic = MonophonicLayer()

    def forward(self, x):
        """Override of the forward method, applying sequentially all the internally
            defined groups of layers.
        Arguments:
            x   The input data batch.
        """
        y = self.fc_net(x)
        y = self.transp_conv_net(y)
        y = self.monophonic(y)
        assert (y <= 1).all(), "Found a value bigger than one"
        return y


# Discriminator Architecture - No minibatch disc.
# class Discriminator(nn.Module):
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
    """This class represent the GAN discriminator, it's an extension of the torch.nn module.
    It is composed by four groups of layers:
        -first group of convolutional layers
        -second group of convolutional layers
        -a batch discrimination layer
        -a group of linear layers
    Activations (LeakyRelu),  batch normalization and dropouts(with probability equal to 0.3)
    are used.

    Attributes:
        conv_net1         The first group of convolutional layers.
        conv_net2         The second group of convolutional layers.
        bd_net            The batch discrimination layers placed before the last layer of the
                          discriminator network.
        fc_net            The group of linear layers.
    """

    def __init__(self, minibatch_B=10, minibatch_C=5):
        """This constructor define the layers groups and set the parameters for the batch
        discrimination layer.
        Arguments:
            minibatch_B   The size of the second component of the tensor T in bd_net.
            minibatch_B   The size of the third component of the tensor T in bd_net.
        """
        super().__init__()

        self.conv_net1 = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(128, 2), stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )

        self.conv_net2 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=77, kernel_size=(1, 4), stride=2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231)
            nn.Flatten(start_dim=1),
        )

        minibatch_A = (
            231  # This number is obtained as the size of the output of conv_net2
        )
        self.bd_net = MinibatchDiscrimination(minibatch_A, minibatch_B, minibatch_C)

        self.fc_net = nn.Sequential(
            nn.Linear(in_features=231 + minibatch_B, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=1),
            # nn.Sigmoid()
        )

    def forward(self, x, feature_out=False):
        """Override of the forward method, applying sequentially all the layers groups.
        Also if the parameter feature_out is True it just return the output of the first
        convolutional layers group (used to compute the feature match loss later in the GAN class)
        Arguments:
            x             The batch for the current forward pass.
            feature_out   The boolean variable to decide what to output
        """
        y = self.conv_net1(x)

        if feature_out:
            return y

        features_flattened = self.conv_net2(y)

        mbd_output = self.bd_net(features_flattened)

        # Cancatenation of the minibatch discrimination computed features with the
        # one of the second group of convolutions along the second dimension.
        combined_features = torch.cat((features_flattened, mbd_output), dim=1)

        output = self.fc_net(combined_features)

        return output


class GAN(L.LightningModule):
    """This class implement the entire GAN architecture, to do this we exploited the lightning
    module which allow to do a lot of operations implicitely.
    Here we defined the training steps, the optimizers for both the generator and the discriminator,
    the forward pass and the losses to use.

    Attributes:
        latent_dim      The dimension of the input noise for the first iteration.
        lr_d            The learning rate of the discriminator adam optimizer.
        lr_g            The learning rate of the generator adam optimizer.
        b1              The first beta coefficient of the adam optimizer.
        b2              The second beta coefficient of the adam optimizer.
        gen_updates     The amount of updates of the generator performed after each iteration.
        dis_updates     The amount of updates of the discriminator performed after each iteration.
        lambda_1        The coefficient of the importance of the data distance between fake and
                        real data.
        lambda_2        The coefficient of the importance of the feature distance between fake and
                        real data.
        minibatch_B     The size of the minibatch discrimination tensor second component.
        minibatch_C     The size of the minibatch discrimination tensor third component.
        generator       The generator instance used in this module.
        discriminator   The discriminator instance used in this module.
        validation_z    Variable used to see intermediate result after each epoch
    """

    def __init__(
        self,
        # Dim of the noise vector used as input in the generator.
        latent_dim: int = 100,
        # Image output shape:
        # channels,
        # width,
        # height,
        # Learning rate (to tune).
        lr_d: float = 0.0002,
        lr_g: float = 0.0002,
        # Adam optimizer params (to tune).
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
        """This constructor initialize all the main elements of the GAN (which where
        listed previously in the GAN class definition).
        """
        # To initialize correctly the superclass.
        super().__init__()

        # Function to save all the hyperparameters passed in the above constructor.
        self.save_hyperparameters()

        # To control manually the optimization step, since in GAN we have to use criteria
        # in optimizing both Generator and discriminator.
        self.automatic_optimization = False

        # Newtorks definitions (Generator + Discriminator).
        # data_shape = (channels, width, height)
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(minibatch_B, minibatch_C)

        # Used to see intermediate outputs after each epoch.
        self.validation_z = torch.randn(10, latent_dim)

    # TO check: __call__ = forward
    def forward(self, x):
        """This function override the forward definition. Here we just compute the
        forward pass of the instance generator."""
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        """This funcrtion is the definition of the adversarial loss in the GAN network.
        Here we decided to use the standard BCEWithLogitsLoss."""
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(y_hat, y)

    def validation_step(
        self,
    ):
        """The validation step is skipped, since we are training a GAN network which should
        not require to do this."""
        pass

    def configure_optimizers(self):
        """This function is used in the GAN class to configure the optimizer for both the
        generator and the discriminator instances.
        In this case we chose the adam optimizer.
        """

        # Applying the weights_init function to all layers.
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Setting the parameters for the optimizers.
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # Defining the optimizers with paramaters.
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(b1, b2)
        )
        return [opt_g, opt_d], []

    # It shuld be on_validation_epoch_end
    def on_train_epoch_end(self):
        """This function defines what should be done at the end of the train epoch
        (when a train epoch end this function is called).
        """

        # Clear ouput.
        display.clear_output(wait=True)

        z = self.validation_z.type_as(self.generator.fc_net[0].weight)

        # Generate images.
        sample_curr = self(z).detach().cpu()

        # Grid dimensions.
        cols = 5
        rows = 2

        # Create the figure.
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2))
        axes = axes.flatten()
        for idx, (ax, img) in enumerate(zip(axes, sample_curr)):
            img_np = img.squeeze().numpy()
            im = ax.imshow(img_np, aspect="auto", origin="lower", cmap="hot")
            ax.set_title(f"#{idx}")
            fig.colorbar(im, ax=ax, label="Velocity")

        # Plot the figure.
        plt.tight_layout()
        plt.show()

    def training_step(self, batch):
        """This function defined whar should be done at each train step. Gets from
        the arguments the batch.
        Arguments:
            batch   The batch used to train in the current step.
        """
        # Batch images (for the discriminator).
        curr = batch  # Ignore the labels.

        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Measure discriminator's ability to classify real from generated samples.

            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # Load on GPU because we created this tensor inside training_loop.
            z = z.type_as(curr)

            # Generate new images.
            self.generated_curr = self(z)

            # Activate Generator optimizer.
            self.toggle_optimizer(optimizer_d)

            # Defining the valid vector.
            # The factor 0.9 is used to perform one side label smoothing.
            valid = torch.ones(curr.size(0), 1) * 0.9
            valid = valid.type_as(curr)

            # First term of discriminator loss.
            real_loss = self.adversarial_loss(self.discriminator(curr), valid)

            # Defining the fake vector.
            fake = torch.zeros(curr.size(0), 1)
            fake = fake.type_as(curr)

            # Second term of discriminator loss.
            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_curr.detach()), fake
            )

            # Total discriminator loss.
            d_loss = real_loss + fake_loss

            # WGAN loss.
            # d_loss = torch.mean(self.discriminator(self.generated_curr.detach())) - torch.mean(self.discriminator(curr))

            # Discriminator training.
            self.log("d_loss", d_loss, prog_bar=True)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            ## WGAN.
            # for p in self.discriminator.parameters():
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

            # First term of the generator loss.
            regularizer_1 = torch.norm(mean_img_from_batch - mean_img_from_g) ** 2

            # TOFIX: should we detach??
            # Computing the features using discriminator forward with just the first
            # convolutional layers group.
            real_features = torch.mean(self.discriminator(curr, feature_out=True))
            fake_features = torch.mean(
                self.discriminator(self.generated_curr, feature_out=True)
            )

            # Second term of the generator loss.
            regularizer_2 = torch.norm(real_features - fake_features) ** 2

            # TO CHOOSE WHICH LOSS TO KEEP
            g_adversarial_loss = self.adversarial_loss(
                self.discriminator(self.generated_curr), valid
            )

            ## Total loss.
            g_loss = (
                self.hparams.lambda_1 * regularizer_1
                + self.hparams.lambda_2 * regularizer_2
                + g_adversarial_loss
            )

            # WGAN.
            # g_loss = -torch.mean(self.discriminator(self.generated_curr))

            # Generator training.
            self.log("g_loss", g_loss, prog_bar=True)  # Log loss.
            self.manual_backward(g_loss)  # Toggle.
            optimizer_g.step()  # Update weights.
            optimizer_g.zero_grad()  # Avoid accumulation of gradients.
            self.untoggle_optimizer(optimizer_g)
