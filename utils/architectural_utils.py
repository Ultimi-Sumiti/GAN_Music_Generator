import torch 
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter



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
    


# Function to concatenate the conditioning 2D of the Conditioner 
def prev_concat(x, cond_x): # cond_x is the output tensor of a certain layer of the Conditioner
    
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

# Function to concatenate the 1-D chord condition in the layers of the Generator ad in input to the Discriminator
# Del pair , in curr dobbiamo estrarre il vettore y degli accordi e passarlo come condizione in tutti ilayer transp del generatore e come input al discriminatore (+ la prima convoluzione secondo limplementazione loro)
def chord_concat(x,y):
    
    x_shapes = x.shape
    y_shapes = y.shape
    # to modify, se non sono tensori è da cambiare questa riga togliendola
    y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])
    
    # Concatenate alogn the 1 dimesnion of the tensors [batch_size * (C_x + C_y) * Height * Weight]
    return torch.cat((x, y2),1)
    


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