import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math

import itertools
from itertools import combinations

from utils import *


"""Agents Modules"""

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
    def __init__(self, n_in, traj_hidden, node_dim, edge_dim, n_extractors ,kernel_size, 
                depth, do_prob=0.3, mode="reinforce"):
        super(Actor, self).__init__()
        self.tcn_encoder = TCNEncoder(n_in, traj_hidden, node_dim, kernel_size, 
                                     depth, do_prob)
        self.mlp_e1 = MLP(n_extractors, edge_dim, edge_dim, do_prob=0)
        self.mlp_n1 = MLP(node_dim+edge_dim, node_dim, node_dim, do_prob)
        self.mlp_e2 = MLP(n_extractors+edge_dim, edge_dim, edge_dim, do_prob=0)
        if mode=="reinforce":
            self.fc_out = nn.Linear(edge_dim, 1)
        else:
            self.fc_out = nn.Linear(edge_dim, 2)
        self.edge_extractors_l1 = nn.ParameterList([Parameter(torch.rand(node_dim)) for i in range(n_extractors)])
        self.edge_extractors_l2 = nn.ParameterList([Parameter(torch.rand(node_dim)) for i in range(n_extractors)])
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send, layer):
        """
        args:
           x: the node vector of trajectories
              shape: [n_batch, n_atoms, node_dim]
        """
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        if layer==1:
           edges = [(senders*D*receivers).sum(-1)/senders.size(-1) for D in self.edge_extractors_l1]
        else:
           edges = [(senders*D*receivers).sum(-1)/senders.size(-1) for D in self.edge_extractors_l2]
        edges = torch.stack(edges)
        edges = edges.permute(1,2,0)
        #shape: [b_batch, n_edges, n_extractors]

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
        edges = self.node2edge(nodes, rel_rec, rel_send, 1)
        #shape: [n_batch, n_edges, n_extractors]
        edges = self.mlp_e1(edges)
        #shape: [n_batch, n_edges, edge_dim]
        nodes_2 = self.edge2node(edges, rel_rec, rel_send)
        #shape: [n_batch, n_nodes, edge_dim]
        nodes_2 = torch.cat([nodes, nodes_2], dim=-1)
        #shape: [n_batch, n_nodes, node_dim+edge_dim]
        nodes_2 = self.mlp_n1(nodes_2)
        #shape: [n_batch, n_nodes, node_dim]
        edges_2 = self.node2edge(nodes_2, rel_rec, rel_send, 2)
        #shape: [n_batch, n_edges, n_extractors]
        edges_2 = torch.cat([edges, edges_2], dim=-1)
        edges_2 = self.mlp_e2(edges_2)
        #shape: [n_batch, n_edges, edge_dim]
        out = self.fc_out(edges_2)
        #shape: [n_batch, n_edges, 2]

        return out
        




class Critic(nn.Module):
    """
    approximate the reward given states and actions
    """
    def __init__(self, n_in, traj_hidden, node_dim, edge_dim, graph_dim ,kernel_size,
                depth, do_prob=0.3):
        super(Critic, self).__init__()
        self.tcn_encoder = TCNEncoder(n_in, traj_hidden, node_dim, kernel_size, 
                                     depth, do_prob)
        self.mlp_e1 = MLP(1, edge_dim, edge_dim)
        self.mlp_n1 = MLP(edge_dim+node_dim+graph_dim, node_dim, node_dim)
        self.mlp_e2 = MLP(edge_dim+node_dim+graph_dim, edge_dim, edge_dim)
        self.mlp_g1 = MLP(edge_dim+node_dim, graph_dim, graph_dim)
        self.mlp_g2 = MLP(edge_dim+node_dim, graph_dim, 1)

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

    def en2graph(self, nodes, edges):
        """
        convert edges and nodes to graph properties
        args:
           nodes: [n_batch, n_nodes, node_dim]
           edges: [n_batch, n_edges, edge_dim]
        """
        nodes_ave = nodes.mean(1)
        edges_ave = edges.mean(1)   
        g = torch.cat([nodes_ave, edges_ave], dim=-1)
        #shape: [n_batch, node_dim+edge_dim]
        return g

    

    def forward(self, trajs, actions, rel_rec, rel_send):
        """
        args:
            trajs: [n_batch, n_nodes, n_timesteps, n_in]
            actions: [n_batch, n_edges, 1]
        """
        nodes = self.tcn_encoder(trajs)
        #shape: [n_batch, n_nodes, node_dim]
        edges = self.mlp_e1(actions)
        #shape: [n_batch, n_edges, edge_dim]
        g1 = self.en2graph(nodes, edges)
        #shape: [n_batch, node_dim+edge_dim]
        g1 = g1.unsqueeze(1) #shape: [n_batch, 1, node_dim+edge_dim]
        g1 = self.mlp_g1(g1)
        # get node property
        #shape: [n_batch, n_nodes, graph_dim]
        nodes1 = self.edge2node(edges, rel_rec, rel_send)
        #shape: [n_batch, n_nodes, edge_dim]
        g1_prime = g1.expand(nodes.size(0), nodes.size(1), g1.size(-1))
        #shape: [n_batch, n_nodes, nodes_dim+edge_dim]
        nodes1 = torch.cat([nodes, nodes1, g1_prime], dim=-1)
        #shape: [n_batch, n_nodes, node_dim+edge_dim+graph_dim]
        nodes1 = self.mlp_n1(nodes1)
        #shape: [n_batch, n_nodes, node_dim]
        edges1 = self.node2edge(nodes1, rel_rec, rel_send)
        #shape: [n_batch, n_edges, node_dim]
        g1_prime = g1.expand(edges.size(0), edges.size(1), g1.size(-1))
        #shape: [n_batch, n_nodes, nodes_dim+edge_dim]
        edges1 = torch.cat([edges, edges1, g1_prime], dim=-1)
        #shape: [n_batch, n_edges, node_dim+edge_dim+graph_dim]
        edges1 = self.mlp_e2(edges1)
        #shape: [n_batch, n_edges, edge_dim]
        g2 = self.en2graph(nodes1, edges1)
        #shape: [n_batch, node_dim+edge_dim]
        g2.unsqueeze(1)
        values = self.mlp_g2(g2)
        #shape: [n_batch, 1]
        values = values.squeeze(1)

        return values
        
      
        

    
"""
Correlation Clustering Modules
"""

def compute_all_clusterings(indices):
    """
    args:
        indices: indices of items
    """
    if len(indices)==1:
        yield [indices]
        return
    first = indices[0]
    for smaller in compute_all_clusterings(indices[1:]):
        # insert "first" in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n]+[[first]+subset]+smaller[n+1:]
        yield [[first]]+smaller


def compute_clustering_score(sims, clustering):
    """
    args:
        sims: similarity matrix
        clustering: list of lists denoting clusters
    """
    score = 0.
    for cluster in clustering:
        if len(cluster)>=2:
            combs = list(combinations(cluster, 2))
            for comb in combs:
                score += sims[comb]
    return score


def merge_2_clusters(current_clustering, indices):
    """
    merge 2 clusters of current clustering
    args:
        current_clustering: list of lists denoting clusters
        indices(tuple): indices of 2 clusters of current clustering
    """
    assert len(current_clustering)>1
    num_clusters = len(current_clustering)
    cluster1 = current_clustering[indices[0]]
    cluster2 = current_clustering[indices[1]]
    merged_cluster = cluster1+cluster2
    new_clustering = [merged_cluster]
    for i in range(num_clusters):
        if i!=indices[0] and i!=indices[1]:
            new_clustering.append(current_clustering[i])
    return new_clustering


def greedy_approximate_best_clustering(sims):
    """
    args:
        sims(numpy ndarray): similarity matrices, shape:[n_atoms, n_atoms]
        current_clustering: a list of lists denoting clusters
        current_score: current clustering score
    """
    num_atoms = sims.shape[0]
    current_cluster_indices = list(range(num_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    current_score = 0.
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    best_clustering = current_clustering
    
    
    while(True):
        #merge 2 clusters hierachically
        
        #if len(current_clustering)==1: #cannot be merged anymore
        #    return current_clustering, current_score
        
        best_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_score = compute_clustering_score(sims, new_clustering)
            delta = new_score-current_score
            if delta>best_delta:
                best_clustering = new_clustering
                best_delta = delta
                current_score = new_score
        if best_delta<=0:
            return best_clustering, current_score
        
        current_clustering = best_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, current_score
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))








