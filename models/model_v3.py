""" This file contains all the code for the creation of a GAN to produce melodies
of 8 bars (16 time steps and 128 notes/semitones).

Here we train the GAN (generator and discriminator) with a dataset of samples made of triplets:
    -one pair of bars (previous and current) 
    -one chord 
Pairs are obtained by processing midi files in the original dataset and converting the pretty_midi objects into 1x128x16 tensors, and also by processing the previous bar chords and selecting the dominant one in the bar time steps. The chord is encoded as a 13x1x1 tensor. 
(The pairs of bars which becomes pairs of tensors are couples of subsequent bars in one of the midi files considered)

The main difference between this model and the previous is that we don't just need to consider the 
relation between the previous chord and the current as a 2D condition on th generation of the current, we also need to consider the relation between the previous chord and the current.

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

# import lightning as L
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


# GENERATOR ARCHITECTHURE
class Generator(nn.Module):
    """This class represent the GAN generator, it's an extension of the torch.nn module.
    It is composed by 4 groups of layers:
        -a group of linear layers.
        -a group of transpose convolutional layers.
        -a group of convolutional layers.
        -a custom monophonic layer.
    Both activations (LeakyRelu) and batch normalization are used.

    Attributes:
        input_size          The size of the input noise for the generator.
        pitch_size          The height of input bars.
        a                   The length of input bars.
        transp_layer_size   The size of the transpose convolutions.
        fc_net              The group of linear layers.
        transp_conv        The group of transpose convolution layers (here these are separated
                            so we have transp_conv1, 2, 3 and 4).
        conv_cond           The group of convolutional layers (here these are separated se we have
                            conv1_cond, 2, 3 and 4)
        monophonic          The final custom Function to avoid vanishing gradients and to
                            get only one note per time step.
    """

    def __init__(self, input_size, w_size=128, a=16):
        """Initialize the layers and input size from argument.
        Arguments:
            a            num Channels of the conditioner cnn layers (num of kernels).
            w_size       dim of the w size of the image, in our case is fixed always to 128.
            input_size   dim of the noise vector takes in input.
        """
        super().__init__()

        # Channels of the X in input, in our case it's = 1.
        self.input_size = input_size
        self.pitch_size = w_size  # Si potrebbe gia mettere 128 tanto è fissa TODO.
        self.y_size = 13
        self.a = a

        self.transp_layer_size = self.a + self.pitch_size + self.y_size

        # Conditioner CNN in questo modelv_3 si può decidere se fare l' iniezione solo nell ultimo layer.
        self.conv1_cond = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=a, kernel_size=(128, 1), stride=1),
            nn.BatchNorm2d(a),
            nn.LeakyReLU(),
        )
        self.conv2_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1, 2), stride=2),
            nn.BatchNorm2d(a),
            nn.LeakyReLU(),
        )

        self.conv3_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1, 2), stride=2),
            nn.BatchNorm2d(a),
            nn.LeakyReLU(),
        )
        self.conv4_cond = nn.Sequential(
            nn.Conv2d(in_channels=a, out_channels=a, kernel_size=(1, 2), stride=2),
            nn.BatchNorm2d(a),
            nn.LeakyReLU(),
        )

        # MLP
        self.fc_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, 1, 2)),
        )

        # Transpose convolution layers.
        self.transp_conv1 = nn.Sequential(
            # Default: padding=0, output_padding=0,  dilation=1
            nn.ConvTranspose2d(
                in_channels=self.transp_layer_size,
                out_channels=w_size,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU(),
        )
        self.transp_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.transp_layer_size,
                out_channels=w_size,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU(),
        )

        self.transp_conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.transp_layer_size,
                out_channels=w_size,
                kernel_size=(1, 2),
                stride=2,
            ),
            nn.BatchNorm2d(w_size),
            nn.LeakyReLU(),
        )

        self.transp_conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.transp_layer_size,
                out_channels=1,
                kernel_size=(w_size, 1),
                stride=1,
            ),
            # nn.BatchNorm2d(1), # TO CHECK TODO
            nn.LeakyReLU(),
        )

        self.monophonic = MonophonicLayer()

    # Override of the forward method
    def forward(self, x):
        """Override of the forward method, applying sequentially all the internally
        defined groups of layers. In this specific case we divide the batch into
        previous and current.
        We apply the convolutional layers to the previous.
        Concatenate the outputs of those layers to the chords and then to the output of
        the linear layers applied to current.
        After that we apply the transpose convolutions to the concatenations done
        previously.
        In the end we apply the monophonic layer to ensure that we have a proper bar.
        Arguments:
            x   The input data batch of triplets.
        """

        # x is the noise, prev_x is the previously generated sample obtained from the pair of bars, y is the chord on which
        # we want to condition our melody generation
        # (curr, prev, y), in this case we feed the conditioner with the prev, while the discriminator with curr
        # and we feed each layer of the transp conv of the generator with y, and also the first and second layer of the discriminator

        # Chord.
        y = x[2]
        # Previous bar.
        prev_x = x[1]
        # Current bar.
        x = x[0]

        # Process the previous generated sample in the Conditioner network.
        # First number is the expected according to the paper, second number the one
        # according to be consistent with the implementation.
        cond1 = self.conv1_cond(prev_x)  #  ([bs, a, 1, 16])
        # print("\nDime cond1", cond1.size())
        cond2 = self.conv2_cond(cond1)  #  ([bs, a, 1, 8])
        # print("Dim cond2", cond2.size())
        cond3 = self.conv3_cond(cond2)  #  ([bs, a, 1, 4])
        # print("Dim cond3", cond3.size())
        cond4 = self.conv4_cond(cond3)  #  ([bs, a, 1, 2])
        # print("Dime cond4", cond4.size())

        # At the end we must have that a + b = c , where b is the dim of channel transpose
        # conv layer, and c is the sum between conditioner layer channel (a) and (b).
        o = self.fc_net(x)  # ([bs, w_size, 1, 2])

        # Concatenate conv4 conditioner with o (:= output tensor of fc_net).
        # print("\no after the MLP", o.size())
        o = chord_concat(o, y)  # ([bs, y_size + w_size, 1, 2])
        # print("\no after the first chord concat", o.size())
        o = prev_concat(o, cond4)  # ([bs, y + a + w_size, 1, 2])
        # print("o after first conditional concat:", o.size())
        o = self.transp_conv1(o)  # ([bs, w_size, 1, 4])

        # Concatenate conv3 conditioner with o (:= output tensor of transp_conv1)
        # print("o after the first transp conv", o.size())
        o = chord_concat(o, y)  # ([bs, y_size + w_size, 1 , 4])
        # print("\no after the second chord concat", o.size())
        o = prev_concat(o, cond3)  # ([bs, y_size + a + w_size, 1, 4])
        # print("o after second conditional concat:", o.size())
        o = self.transp_conv2(o)  # ([bs, w_size, 1, 8])

        # Concatenate conv2 conditioner with o (:= output tensor of transp_conv2)
        # print("o after the second transp conv", o.size())
        o = chord_concat(o, y)  # ([bs, y_size + w_size, 1 , 8])
        # print("\no after the third chord concat", o.size())
        o = prev_concat(o, cond2)  # ([bs, y_size + a + w_size, 1, 8])
        # print("o after third conditional concat:", o.size())
        o = self.transp_conv3(o)  ## ([bs, w_size, 1, 16])

        # Concatenate conv1 conditioner with o (:= output tensor of transp_conv3)
        # print("o after the third transp conv", o.size())
        o = chord_concat(o, y)  # ([bs, y_size + w_size, 1 , 16])
        # print("\no after the fourth chord concat", o.size())
        o = prev_concat(o, cond1)  # ([bs, y_size + a + w_size, 1, 16])
        # print("o after fourth conditional concat:", o.size())
        o = self.transp_conv4(o)  ## ([bs, 1, 128, 16])

        # print("o after last transp conv (4)", o.size())

        o = self.monophonic(o)  # ([bs, 1, 128, 16])
        # print("\no after monophonic", o.size())
        # assert (o <= 1).all(), "Found a value bigger than one"
        return o


# DISCRIMINATOR ARCHITECHTURE
class Discriminator(nn.Module):
    """This class represent the GAN discriminator, it's an extension of the torch.nn module.
    It is composed by three/four groups of layers:
        -first group of convolutional layers.
        -second group of convolutional layers.
        -a batch discrimination layer (not always depending on one parameter).
        -a group of linear layers.
    Activations (LeakyRelu),  batch normalization and dropouts(with probability equal to 0.3)
    are used.

    Attributes:
        apply_mbd         Boolean variable to appply batch discrimination if True.
        conv_net1         The first group of convolutional layers.
        conv_net2         The second group of convolutional layers.
        bd_net            The batch discrimination layers placed before the last layer of the
                          discriminator network.
        fc_net            The group of linear layers.
        y_size            The size of the chord tensor.
    """

    def __init__(self, apply_mbd=False, mbd_B_dim=10, mbd_C_dim=5):
        """This constructor define the layers groups and set the parameters for the batch
        discrimination layer if apply_mbd is True, also this sets the y_size.
        Arguments:
            apply_mbd     Boolean variable to appply batch discrimination if True.
            minibatch_B   The size of the second component of the tensor T in bd_net.
            minibatch_B   The size of the third component of the tensor T in bd_net.
        """
        super().__init__()

        # True if one want to use minibatch discrimination.
        self.apply_mbd = apply_mbd
        self.y_size = 13

        self.conv_net1 = nn.Sequential(
            # Default: padding=0,  dilation=1
            nn.Conv2d(
                in_channels=1 + self.y_size,
                out_channels=14,
                kernel_size=(128, 2),
                stride=2,
            ),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )

        self.conv_net2 = nn.Sequential(
            nn.Conv2d(
                in_channels=14 + self.y_size,
                out_channels=77,
                kernel_size=(1, 3),
                stride=2,
            ),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            # Reshape torch.rand(N, 77, 1 ,3) --> torch.rand(N, 231)
            nn.Flatten(start_dim=1),
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
                # nn.Sigmoid()
            )

        # No minibatch discrimination is applied.
        else:
            self.fc_net = nn.Sequential(
                nn.Linear(in_features=231, out_features=1024),
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
            feature_out   The boolean variable to decide what to output.
        """

        # x is the noise, y is the chord on which
        # we want to condition our melody generation
        # (curr, y), in this case we feed the conditioner with the prev, while the discriminator with curr
        # and we feed each layer of the transp conv of the generator with y, and also the first and second layer of the discriminator

        # Chords.
        y = x[1]
        # Previous bars.
        x = x[0]

        o = chord_concat(x, y)  # ([bs, + y_size + 1 , 128, 16])
        # print("\no after the first chord concat", o.size())
        o = self.conv_net1(o)  # ([bs, + y_size + 1 , 1, 8])
        # print("o after the first conv", o.size())

        # Needed for the feature match loss in GAN training.
        if feature_out:
            return o

        # If we apply minibatch discrimination.
        if self.apply_mbd:
            o = chord_concat(
                o, y
            )  # Da controllare nel paper non e scritto nell implementazione loro si TODO    # ([bs, + y_size + 14 , 1, 8])
            # print("\no after the second chord concat (MB discriminator active)", o.size())
            features_flattened = self.conv_net2(o)  # ([bs, 231])
            # print("o after the second conv (MB discriminator active)", features_flattened.size())
            mbd_output = self.bd_net(features_flattened)

            combined_features = torch.cat((features_flattened, mbd_output), dim=1)

            o = self.fc_net(combined_features)  # ([bs, 1])
            # print("\no after the MLP", o.size())

        # No minibatch discrimination is applied.
        else:
            o = chord_concat(
                o, y
            )  # Da controllare nel paper non e scritto nell implementazione loro si TODO    # ([bs, + y_size + 14 , 1, 8])
            # print("\no after the second chord concat ", o.size())
            o = self.conv_net2(o)  # ([bs, 231])
            # print("o after the second conv", o.size())

            o = self.fc_net(o)  # ([bs, 1])
            # print("\no after the MLP", o.size())

        return o


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
        validation_z    Variable used to see intermediate result after each epoch.
        a               Num Channels of the conditioner cnn layers (num of kernels).
    """

    # Constructor.
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
        lambda_1=0.1,
        lambda_2=1,
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
        # To initialize correctly the superclass
        super().__init__()

        # Function to save all the hyperpars passed in constructor.
        self.save_hyperparameters()

        # To control manually the optimization step, since in GAN we have to use criteria
        # in optimizing both Generator and discriminator.
        self.automatic_optimization = False

        # Newtorks definition(Generator + Discriminator).
        # data_shape = (channels, width, height).
        self.generator = Generator(latent_dim, a=a)
        self.discriminator = Discriminator(
            apply_mbd=apply_mbd, mbd_B_dim=mbd_B_dim, mbd_C_dim=mbd_C_dim
        )

    # Forward step computed.
    def forward(self, x):
        """This function override the forward definition. Here we just compute the
        forward pass of the instance generator."""
        return self.generator(x)

    # Loss of the Adversarial game.
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

    # Gan training algorithm.
    def training_step(self, batch):
        """This function defines what should be done at the end of the train epoch
        (when a train epoch end this function is called).
        """

        # Batch images.
        prev, curr, y = batch

        # Define the optimizers.
        optimizer_g, optimizer_d = self.optimizers()

        ### DISCRIMINATOR ####
        for _ in range(self.hparams.dis_updates):
            # Sample noise for the generator.
            z = torch.randn(curr.shape[0], self.hparams.latent_dim)
            # Load on GPU because we created this tensor inside training_loop.
            z = z.type_as(curr)

            # Generate new images.
            self.generated_curr = self.generator((z, prev, y))

            # Activate Generator optimizer.
            self.toggle_optimizer(optimizer_d)

            # Defining the valid tensor
            # One-sided label smoothing
            valid = torch.ones(curr.size(0), 1) * 0.9
            valid = valid.type_as(curr)

            # maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator((curr, y))
            fake_pred = self.discriminator((self.generated_curr.detach(), y))

            # First term of discriminator loss.
            real_loss = self.adversarial_loss(real_pred, valid)

            # Defining the fake tensor.
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

            # Generate images.
            # this could be also non self (modify)
            self.generated_curr = self.generator((z, prev, y))

            # Log sampled images.
            # TODO

            # Ground truth result (ie: all fake)
            valid = torch.ones(curr.size(0), 1)
            # Load on GPU because we created this tensor inside training_loop.
            valid = valid.type_as(curr)

            # maximize log(D(G(z))) + lamda1 * ||mean_img_from_batch - mean_img_from_g||_2 + lamda2 * ||real_features - fake_features||_2
            # Compute the loss function with the application of Feature Matching techinque
            mean_img_from_batch = torch.mean(curr, 0)
            mean_img_from_g = torch.mean(self.generated_curr, 0)

            # First term of the generator loss.
            regularizer_1 = torch.norm(mean_img_from_batch - mean_img_from_g) ** 2

            # Computing the features using discriminator forward with just the first
            # convolutional layers group.
            real_features = torch.mean(
                self.discriminator((curr, y), feature_out=True), 0
            )
            fake_features = torch.mean(
                self.discriminator((self.generated_curr, y), feature_out=True), 0
            )

            # Second term of the generator loss.
            regularizer_2 = torch.norm(real_features - fake_features) ** 2

            # TO CHOOSE WHICH LOSS TO KEEP
            g_adversarial_loss = self.adversarial_loss(
                self.discriminator((self.generated_curr, y)), valid
            )

            # Total Generator loss.
            g_loss = (
                self.hparams.lambda_1 * regularizer_1
                + self.hparams.lambda_2 * regularizer_2
                + g_adversarial_loss
            )

            # Generator training iteration step.
            self.log("g_loss", g_loss, prog_bar=True)  # Log loss.
            self.manual_backward(g_loss)  # Toggle.

            # Compute gradient norm.
            grad_norm = get_gradient_norm(self.generator)
            self.log("grad_norm", grad_norm, prog_bar=True)

            optimizer_g.step()  # Update weights.
            optimizer_g.zero_grad()  # Avoid accumulation of gradients.
            self.untoggle_optimizer(optimizer_g)


# Tester to see if the dimensions and matches between various layers are correct
if __name__ == "__main__":
    # Note we have to reason considering the w_size as fixed to 128 to modify the other params of the net
    # Generator(#dim of noise, #w_size of the image, in our case is always 128, #num of kernels of the conditioner layers)
    model_g = Generator(100, 128, 25)
    model_d = Discriminator(True)
    model_g.eval()  # Set to evaluation mode to test, othewise BatchNorm can't worktopic
    model_d.eval()  # Set to evaluation mode to test, othewise BatchNorm can't work

    # Note, is IMPORTANT that conv_x and z have the same BATCH_SIZE
    z = torch.randn(5, 100)
    # print(z)
    # print(z.size())

    # For the 2-D previous bar condition
    conv_x = torch.zeros(5, 1, 128, 16)
    # print(conv_x)
    # For the 1-D chord condition
    y = torch.zeros(5, 13, 1, 1)
    # print(y)

    # ------- single GENERATOR -------
    # For how is implemented the dataset 2 we need to pass noise + prev + y as a triplet for the generator
    triplet = (z, conv_x, y)
    # Call to forward of the generator
    print("\nOutput Generator")
    o = model_g(triplet)
    print(o)

    # ------- single DISCRIMINATOR -------
    # For how is implementet the dataset and the discriminator we need to pass fake_img (output of the generator) + y 1-D Condition vector
    # as a pair to the discriminator
    print("\nOutput Discriminator")
    pair = (o, y)
    o = model_d(pair)
    print(o)

    # ------- GAN -------
    print("\nOutput GAN")
    model = GAN()

    o = model(triplet)
    print(o)
