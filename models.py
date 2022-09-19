import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import *



class CausalConv1d(nn.Module):
    """
    causal conv1d layer
    return the sequence with the same length after
    1D causal convolution
    Input: [B, in_channels, L]
    Output: [B, out_channels, L]
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation*(kernel_size-1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=self.padding, dilation=dilation)
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        if self.kernel_size==1:
            return x
        return x[:,:,:-self.padding]




class GatedCausalConv1d(nn.Module):
    """
    Gated Causal Conv1d Layer
    h_(l+1)=tanh(Wg*h_l)*sigmoid(Ws*h_l)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(GatedCausalConv1d, self).__init__()
        self.convg = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation) #gate
        self.convs = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation)
        
    def forward(self, x):
        return torch.sigmoid(self.convg(x))*torch.tanh(self.convs(x))



class GatedResCausalConvBlock(nn.Module):
    """
    Gated Residual Convolutional block
    """     
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(GatedResCausalConvBlock, self).__init__()
        self.conv1 = GatedCausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = GatedCausalConv1d(n_out, n_out, kernel_size, dilation*2)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return x







class TCNEncoder(nn.Module):
    """
    compute the vector representation given the trajectories
    """
    def __init__(self, n_in, c_hidden, c_out, kernel_size,
                 depth):
        super(TCNEncoder, self).__init__()
        res_layers = []#gated residual TCN layers
        for i in range(depth):
            in_channels = n_in if i==0 else c_hidden
            res_layers += [GatedResCausalConvBlock(in_channels, c_hidden, kernel_size,
                                              dilation=2**(2*i))]
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(c_hidden, c_out,kernel_size=1)
        self.conv_attention = nn.Conv1d(c_hidden,1,kernel_size=1)

    def forward(self, inputs):
        """
        args:
           inputs: [batch_size, num_atoms, num_timesteps, num_features]
        return latents of the trajectories
        """
        x = inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2), inputs.size(3))
        #shape: [total_trajectories, num_timesteps, n_in]
        x = x.transpose(-2,-1)
        #shape: [total_trajectories, n_in, num_timesteps]
        x = self.res_blocks(x)
        #shape: [total_trajectories, c_hidden, num_timesteps]
        pred = self.conv_predict(x)
        attention = F.softmax(self.conv_attention(x), dim=-1)
        out = (pred*attention).mean(dim=2) #shape: [total_trajectories, c_out]
        out = out.view(inputs.size(0), inputs.size(1), -1)
        #shape: [n_batch, n_atoms, c_out]

        return out




class Actor(nn.Module):
    """
    A GNN actor maps state to action
    """
    




class Critic(nn.Module):
    """
    evaluate the Q(s,a) value
    """