from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import *



class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
         n_in: #units of input layers
         n_hid: #units of hidden layers
         n_out: #units of output layers
         do_prob: dropout probability
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def batch_norm(self, inputs):
        """
        inputs.size(0): batch size
        inputs.size(1): number of channels
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)
    
    
    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        #print(type(inputs))
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)





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
                 depth, do_prob=0.3):
        super(TCNEncoder, self).__init__()
        res_layers = []#gated residual TCN layers
        for i in range(depth):
            in_channels = n_in if i==0 else c_hidden
            res_layers += [GatedResCausalConvBlock(in_channels, c_hidden, kernel_size,
                                              dilation=2**(2*i))]
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(c_hidden, c_out,kernel_size=1)
        self.conv_attention = nn.Conv1d(c_hidden,1,kernel_size=1)
        self.dropout_prob = do_prob
        

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
        x = F.dropout(x, self.dropout_prob, training=self.training)
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
    def __init__(self, n_in, traj_hidden, node_dim, edge_dim ,kernel_size, 
                depth, do_prob=0.3):
        super(Actor, self).__init__()
        self.tcn_encoder = TCNEncoder(n_in, traj_hidden, node_dim, kernel_size, 
                                     depth, do_prob)
        self.mlp_e1 = MLP(node_dim, edge_dim, edge_dim, do_prob)
        self.mlp_n1 = MLP(node_dim+edge_dim, node_dim, node_dim, do_prob)
        self.mlp_e2 = MLP(node_dim+edge_dim, edge_dim, edge_dim, do_prob)
        self.fc_out = nn.Linear(edge_dim, 1)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send):
        """
        args:
           x: the node vector of trajectories
              shape: [n_batch, n_atoms, node_dim]
        """
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = receivers*senders
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        """
        args:
            x: the edge vector 
                shape: [n_batch, n_edges, edge_dim]
        """
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)

    def forward(self, inputs, rel_rec, rel_send):
        nodes = self.tcn_encoder(inputs)
        #shape: [n_batch, n_nodes, node_dim]
        edges = self.node2edge(nodes, rel_rec, rel_send)
        #shape: [n_batch, n_edges, node_dim]
        edges = self.mlp_e1(edges)
        #shape: [n_batch, n_edges, edge_dim]
        nodes_2 = self.edge2node(edges, rel_rec, rel_send)
        #shape: [n_batch, n_nodes, edge_dim]
        nodes_2 = torch.cat([nodes, nodes_2], dim=-1)
        #shape: [n_batch, n_nodes, node_dim+edge_dim]
        nodes_2 = self.mlp_n1(nodes_2)
        #shape: [n_batch, n_nodes, node_dim]
        edges_2 = self.node2edge(nodes_2, rel_rec, rel_send)
        #shape: [n_batch, n_edges, node_dim]
        edges_2 = torch.cat([edges, edges_2], dim=-1)
        edges_2 = self.mlp_e2(edges_2)
        #shape: [n_batch, n_edges, edge_dim]
        actions = self.fc_out(edges_2)
        #shape: [n_batch, n_edges, 1]

        return actions 
        





class Critic(nn.Module):
    """
    evaluate the Q(s,a) value
    """