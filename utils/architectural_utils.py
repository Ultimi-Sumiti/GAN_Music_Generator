""" In this file we placed all the function that have architectural roles. """

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter


def get_gradient_norm(model, norm_type=2):
    """This function computes the gradient norm of the model params.
    Arguments:
        model       The entire model from which we extract the params
        norm_type   Which kind of norm that we need to compute.
    """
    total_norm = 0.0
    # Iterating the paramters.
    for p in model.parameters():
        # Here we consider params with a non null gradient.
        if p.grad is not None:
            # Computing the single norm of the param and summing to the total.
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    # Computing the selected norm type of the total.
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def weights_init(m):
    """This function is used to initialize the weights of the model.
    Arguments:
        m   The model.
    """
    # Getting all type of layers and initializing the values of the weights
    # using the xavier uniform distribution.
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)


class MonophonicSTE(Function):
    """This class is an extension fo the Function object, so we define a custom type of
    activation function to select proper bars and not considering this in the gradient computation.
    (Strategy to bypass the problem of the threshold wich causes non differentiable function)
    """

    @staticmethod
    def forward(ctx, x):
        """This function is used to perform the forward pass.
        Arguments:
            x   The input tensor.
        """
        # x: (N, C, F, T) for example.
        # Find the max along the feature axis (dim=2).
        _, max_idx = x.max(dim=2, keepdim=True)
        # Create a hard one‐hot tensor.
        y_hard = torch.zeros_like(x)
        y_hard.scatter_(2, max_idx, 1.0)
        # Save nothing for backward, since gradient is identity.
        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        """This function implements the Straight‐through: pass the gradient
        through unchanged.
        """
        return grad_output


class MonophonicLayer(nn.Module):
    """This class is a torch module with a forward pass which adopt the MonophonicSTE
    function to get the Straight‐through.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """This function override the forward torch Module function.
        during forward you get the hard mask, but in backward the grad w.r.t. x
        is grad_output (identity).
        Arguments:
            x   Input tensor.
        """
        return MonophonicSTE.apply(x)


def prev_concat(
    x, cond_x
):  # cond_x is the output tensor of a certain layer of the Conditioner
    """Function used in the conditioner GAN to concatenate the conditioning
    2D of the Conditioner.
    Arguments:
        x        First input tensor.
        cond_x   seconS input tensor.


    """

    x_shapes = x.shape
    cond_x_shapes = cond_x.shape
    # Here we are expecting that the shapes from the third to be equal.
    if x_shapes[2:] == cond_x_shapes[2:]:
        # Performing expansion on the cond_x to get same shapes in first third and
        # fourth dimensions.
        new_cond_x = cond_x.expand(
            x_shapes[0], cond_x_shapes[1], x_shapes[2], x_shapes[3]
        )
        # The size of the new tensor will be [batch_size * (C_x + C_cond_x) * Height * Weight].
        return torch.cat((x, new_cond_x), 1)

    else:
        # If coordinated from the third on, are not equal we have a problem.
        print("Problem in the match of x_shapes[2:] with new_cond_x[2:]")
        print(x_shapes[2:])
        print(cond_x_shapes[2:])


# Del pair , in curr dobbiamo estrarre il vettore y degli accordi e passarlo come condizione in tutti ilayer transp del generatore e come input al discriminatore (+ la prima convoluzione secondo limplementazione loro)
def chord_concat(x, y):
    """Function to concatenate the 1-D chord condition in the layers of the Generator
    in input to the Discriminator.
    Arguments.
        x   Bar tensor.
        y   Chord tensor.

    """

    x_shapes = x.shape
    y_shapes = y.shape
    # to modify, se non sono tensori è da cambiare questa riga togliendola
    # Performing expansion on the chord tensor, keeping only the second dimension.
    y2 = y.expand(x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3])

    # Concatenate along the 1 dimension of the tensors [batch_size * (C_x + C_y) * Height * Weight].
    return torch.cat((x, y2), 1)


class MinibatchDiscrimination(nn.Module):
    """This class is a version of the torch Module. It implements the logic behind the
    minibatch discrimination.
    Attributes:
        feat_num   This is the amount of features considered in input.
        mat_cols   This is the dimension of the second coordinate of the tensor.
                   (can be thought as the amount columns in the ouput matrix for each feature)
        mat_rows   This is the dimension of the third coordinate of the tensor.
                   (can be thought as the amount rows in the ouput matrix for each feature)
        T          This is the tensor used to perform the minibatch discrimination.
    """

    def __init__(self, A, B, C):
        """This constructor initialize the instance attributes given in the arguments.
        Also here we build and initialize the tensor used for the minibatch discrimination.
        """
        super(MinibatchDiscrimination, self).__init__()
        self.feat_num = A
        self.mat_cols = B
        self.mat_rows = C
        # Here the tensor is set as a tensor Parameter so that it is considered by the
        # backpropagation process.
        self.T = Parameter(torch.Tensor(A, B, C))
        # Initialize the tensor.
        self.reset_parameters()

    def forward(self, x):
        """This function overrides the torch Module. Here we performed all the operation
        to get minibatch discrimination based on the T torch tensor values.
        Arguments:
            x   Torch tensors.
        """
        N = x.size(0)
        # Step 1: Reshape T for matmul: (A, B*C).
        T_reshaped = self.T.view(self.feat_num, self.mat_cols * self.mat_rows)

        # Step 2: Perform matrix multiplication: (A, B*C) -> (N, B*C).
        matmul_output = x.matmul(T_reshaped)

        # Step 3: Reshape the matmul output to (N, B, C).
        M = matmul_output.view(N, self.mat_cols, self.mat_rows)

        # Computing the difference between the tensor product (in the score function).
        M_diff = M.unsqueeze(1) - M.unsqueeze(0)

        # Computing the L1 norm for the exponential in the score function.
        M_diff_mod = torch.sum(torch.abs(M_diff), dim=3)

        # Apply the actual exponential for computing the score:
        c = torch.exp(-M_diff_mod)

        # To compute the output we need to compute the sum along the first
        # dimension of the matrices.
        scores = torch.sum(c, dim=1)

        # Now we need to compute the autocorrelation and substruct ot the scores .
        diag_elements = c[
            torch.arange(N, device=x.device), torch.arange(N, device=x.device)
        ]

        # Getting the scores without the autocorrelation.
        out = scores - diag_elements

        return out

    def reset_parameters(self):
        """This function is thought to initialize the internal tensor T.
        This process is done by using a uniform distribution.
        """
        # Computing the standard deviation.
        stddev = 1.0 / torch.sqrt(torch.tensor(self.feat_num, dtype=torch.float32))
        # Intializing the T tensor using uniform distribution.
        self.T.data.uniform_(-stddev, stddev)
