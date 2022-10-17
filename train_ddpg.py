import os
import json
import pickle

import time
import datetime

import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from models import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="Disables CUDA training.")
parser.add_argument("--noise-std", type=float, default=0.4,
            help="Standard deviation of the exploration noise.")
parser.add_argument("--replay-size", type=int, default=10000,
            help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=128,
            help="batch size of replay.")
parser.add_argument("--epochs", type=int, default=500,
                    help="Number of epochs to train.")
parser.add_argument("--lr-actor", type=float, default=0.001,
            help="learning rate of the actor network.")
parser.add_argument("--lr-critic", type=float, default=0.003,
            help="learning rate of the critic network.")
parser.add_argument("--save-folder", type=str, default="logs/ddpg",
            help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
            help="Where to load trained model.")
parser.add_argument("--train-actor-from", type=int, default=10,
                    help="train actor from episode n.")
parser.add_argument("--n-in", type=int, default=2, 
                    help="Dimension of input.")
parser.add_argument("--traj-hidden", type=int, default=32, 
                    help="hidden dimension of trajectories.")
parser.add_argument("--node-dim", type=int, default=64, 
                    help="Dimension of node network.")
parser.add_argument("--edge-dim", type=int, default=64,
                    help="Dimension of edge network.")
parser.add_argument("--n-extractors", type=int, default=32,
                    help="Number of edge feature extractors.")
parser.add_argument("--kernel-size", type=int, default=5, 
                    help="kernel size of temporal convolution.")
parser.add_argument("--depth", type=int, default=1, 
                    help="depth of TCN blocks.")
parser.add_argument("--dropout", type=float, default=0.3,
                    help="dropout probability.")
parser.add_argument("--max-nodes", type=int, default=30,
                    help="maximal number of nodes.")
parser.add_argument("--suffix", type=str, default="zara01",
                    help="Suffix for training data ")
parser.add_argument("--split", type=str, default="split00",
                    help="Split of the dataset.")
parser.add_argument("--timesteps", type=int, default=15,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=500,
                    help="After how many epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)


log = None 
#save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/{}_{}/".format(args.save_folder, timestamp, args.suffix+args.split)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    actor_file = os.path.join(save_folder, "actor.pt")
    actor_target_file = os.path.join(save_folder, "actor_target.pt")
    critic_file = os.path.join(save_folder, "critic.pt")

    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    pickle.dump({"args":args}, open(meta_file, 'wb'))

else:
    print("WARNING: No save_folder provided!"+
          "Testing (within this script) will throw an error.")

# Load data
data_folder = os.path.join("data/pedestrian/", args.suffix)
data_folder = os.path.join(data_folder, args.split)

with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)

lr_actor = args.lr_actor
lr_critic = args.lr_critic
train_actor_from = args.train_actor_from
n_in = args.n_in
traj_hidden = args.traj_hidden
node_dim = args.node_dim
edge_dim = args.edge_dim
n_extractors = args.n_extractors
kernel_size = args.kernel_size
depth = args.depth
do_prob = args.dropout

actor = Actor(n_in, traj_hidden, node_dim, edge_dim, 
             n_extractors, kernel_size, depth, do_prob,
             mode="reinforce")

critic = Critic(n_in, traj_hidden, node_dim, edge_dim,
               n_extractors, kernel_size, depth, do_prob)

if args.cuda:
    actor.cuda()
    critic.cuda()


optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
scheduler_actor = lr_scheduler.StepLR(optimizer_actor, step_size=args.lr_decay, gamma=args.gamma)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
scheduler_critic = lr_scheduler.StepLR(optimizer_critic, step_size=args.lr_decay, gamma=args.gamma)
loss_critic = nn.SmoothL1Loss()

replay_buffer = ReplayBuffer(max_size=args.replay_size, batch_size=args.batch_size)
ou_noise = OUActionNoise(0, args.noise_std)


def act(state, rel_rec, rel_send):
    """
    use the actor net to compute the action
    given the state
    args:
        state: the current state, a torch tensor with the shape:
           [n_batch=1, n_nodes, n_timesteps, n_in]
    """
    actor.eval()
    actions = actor(state, rel_rec, rel_send)
    #shape: [n_batch=1, n_edges, 1]
    return actions

def update_params(episode):
    """
    update the parametres of actor and critic
    """
    # sample transitions from the replay buffer and train the critic network
    critic.train()
    batch = replay_buffer.get_minibatch()
    batch_trans = list(map(list, zip(*batch)))
    states = batch_trans[0]
    actions = batch_trans[1]
    rewards = batch_trans[2]

    training_indices = np.arange(len(states))
    np.random.shuffle(training_indices)
    optimizer_critic.zero_grad()
    idx_count = 0

    for idx in training_indices:
        state = states[idx]
        action = actions[idx]
        reward = torch.tensor(rewards[idx])
        num_nodes = state.size(1)
        rel_rec, rel_send = create_edgeNode_relation(num_nodes, self_loops=False)

        if args.cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()

        state, action = state.float(), action.float()
        Q_predicted = torch.tanh(critic(state, action, rel_rec, rel_send))
        #train critic network
        loss = loss_critic(Q_predicted, reward)
        loss = loss/args.batch_size
        loss.backward()
        idx_count += 1

        if idx_count%args.batch_size==0:
            optimizer_critic.step()
            scheduler_critic.step()
            optimizer_critic.zero_grad()

    
    # train the actor network to maximize the Q-values
    if episode >= train_actor_from:
        actor.train()
        critic.eval()
        idx_count = 0
        optimizer_actor.zero_grad()

        for idx in training_indices:
            state = states[idx]
            num_nodes = state.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_nodes, self_loops=False)
            if args.cuda:
                state = state.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
            action = torch.tanh(actor(state, rel_rec, rel_send))
            Q = torch.tanh(critic(state, action, rel_rec, rel_send))
            #shape: [1, 1]
            actor_loss = -Q/args.batch_size
            actor_loss.backward()
            idx_count += 1

            if idx_count%args.batch_size==0:
                optimizer_actor.step()
                scheduler_actor.step()
                optimizer_actor.zero_grad()
                
            

            




        
        
    







    








