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

from sknetwork.topology import get_connected_components

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="Disables CUDA training.")
parser.add_argument("--noise-std", type=float, default=0.3,
            help="Standard deviation of the exploration noise.")
parser.add_argument("--noise-std-good", type=float, default=0.5,
            help="Standard deviation of the good exploration noise.")
parser.add_argument("--replay-size", type=int, default=10000,
            help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=128,
            help="batch size of replay.")
parser.add_argument("--epochs", type=int, default=500,
                    help="Number of epochs to train.")
parser.add_argument("--lr-actor", type=float, default=0.0003,
            help="learning rate of the actor network.")
parser.add_argument("--lr-critic", type=float, default=0.0005,
            help="learning rate of the critic network.")
parser.add_argument("--save-folder", type=str, default="logs/ddpg",
            help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
            help="Where to load trained model.")
parser.add_argument("--train-actor-from", type=int, default=5,
                    help="train actor from episode n.")
parser.add_argument("--n-in", type=int, default=2, 
                    help="Dimension of input.")
parser.add_argument("--traj-hidden", type=int, default=32, 
                    help="hidden dimension of trajectories.")
parser.add_argument("--node-dim", type=int, default=128, 
                    help="Dimension of node network.")
parser.add_argument("--edge-dim", type=int, default=128,
                    help="Dimension of edge network.")
parser.add_argument("--n-extractors", type=int, default=64,
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
parser.add_argument("--beta", type=float, default=0.1,
                    help="threshold of SmoothL1Loss.")
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
loss_critic = nn.SmoothL1Loss(args.beta)

replay_buffer = ReplayBuffer(max_size=args.replay_size, batch_size=args.batch_size)
ou_noise = OUActionNoise(0, args.noise_std)
ou_noise_good = OUActionNoise(0, args.noise_std_good)


def act(state, rel_rec, rel_send):
    """
    use the actor net to compute the action
    given the state
    args:
        state: the current state, a torch tensor with the shape:
           [n_batch=1, n_nodes, n_timesteps, n_in]
    """
    actions = torch.tanh(actor(state, rel_rec, rel_send))
    #shape: [n_batch=1, n_edges, 1]
    return actions


def explore(episode):
    """
    get more experiences
    """
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    with torch.no_grad():
        for idx in training_indices:
            example = examples_train[idx]
            label = labels_train[idx]
            label = torch.diag_embed(label).float()


            example = example.unsqueeze(0)
            num_nodes = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_nodes, self_loops=False)
            rel_rec, rel_send = rel_rec.float(), rel_send.float()
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()

            label_converted = torch.matmul(rel_send.t(),
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            #shape: [n_nodes, n_nodes]
            if label_converted.sum()==0:
                gID = list(range(label_converted.shape[1]))
            else:
                gID = list(get_connected_components(label_converted))
            gID = indices_to_clusters(gID)

            actions = act(example, rel_rec, rel_send)
            actions = add_symmetric_noise(actions, num_nodes ,rel_rec, rel_send, ou_noise)
            sims = actions_to_sims(actions, rel_rec, rel_send)
            g_predict,_ = greedy_approximate_best_clustering(sims)
            recall, precision, F1 = compute_groupMitre(gID, g_predict)
            reward = compute_reward_f1(F1)

            replay_buffer.add_experience(example, actions, reward)


def explore_good(episode):
    """explore good fake actions"""
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    for idx in training_indices:
        example = examples_train[idx]
        num_nodes = example.size(0)
        #create mask
        #create mask
        mask = np.random.choice([0,1],size=(num_nodes,num_nodes),p=[0.2,0.8])
        mask = 0.5*(mask+mask.T)
        off_diag_idx = np.ravel_multi_index(np.where(np.ones((num_nodes,num_nodes))-np.eye(num_nodes)),
                                   [num_nodes,num_nodes])
        mask = torch.from_numpy(mask.reshape(-1)[off_diag_idx]).float()
        mask = mask.unsqueeze(0).unsqueeze(-1)

        rel_rec, rel_send = create_edgeNode_relation(num_nodes)
        label = labels_train[idx]
        label_diag = torch.diag_embed(label).float()
        label_converted = torch.matmul(rel_send.t(),
                              torch.matmul(label_diag, rel_rec))
        label_converted = label_converted.cpu().detach().numpy()
        #shape: [n_nodes, n_nodes]
        if label_converted.sum()==0:
            gID = list(range(label_converted.shape[1]))
        else:
            gID = list(get_connected_components(label_converted))
        gID = indices_to_clusters(gID)
        
        label_action = label.unsqueeze(0).unsqueeze(-1)*mask
        if random.random()<0.5: #whether convert the range to [-1,1]
            label_action = 2*(label_action-0.5)
        actions = add_symmetric_noise(label_action, num_nodes, rel_rec, rel_send, ou_noise_good)
        sims = actions_to_sims(actions, rel_rec, rel_send)
        g_predict, _ = greedy_approximate_best_clustering(sims)
        recall, precision, F1 = compute_groupMitre(gID, g_predict)
        reward = compute_reward_f1(F1)

        replay_buffer.add_experience(example.unsqueeze(0), actions, reward)




def explore_random(episode):
    """explore random actions."""
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    for idx in training_indices:
        example = examples_train[idx]
        num_nodes = example.size(0)
        actions = np.random.uniform(-1,1, size=(num_nodes, num_nodes))
        actions = 0.5*(actions+actions.T)
        off_diag_idx = np.ravel_multi_index(np.where(np.ones((num_nodes,num_nodes))-np.eye(num_nodes)),
                                   [num_nodes,num_nodes])
        actions = torch.from_numpy(actions.reshape(-1)[off_diag_idx]).float()
        actions = actions.unsqueeze(0).unsqueeze(-1)

        rel_rec, rel_send = create_edgeNode_relation(num_nodes)
        label = labels_train[idx]
        label_diag = torch.diag_embed(label).float()
        label_converted = torch.matmul(rel_send.t(),
                              torch.matmul(label_diag, rel_rec))
        label_converted = label_converted.cpu().detach().numpy()
        #shape: [n_nodes, n_nodes]
        if label_converted.sum()==0:
           gID = list(range(label_converted.shape[1]))
        else:
            gID = list(get_connected_components(label_converted))

        gID = indices_to_clusters(gID)
        sims = actions_to_sims(actions, rel_rec, rel_send)
        g_predict, _ = greedy_approximate_best_clustering(sims)
        recall, precision, F1 = compute_groupMitre(gID, g_predict)
        reward = compute_reward_f1(F1)
        replay_buffer.add_experience(example.unsqueeze(0), actions, reward)



            

def validate(episode):
    """
    validate actor and critic on the validation data sets
    """
    actor.eval()
    critic.eval()
    loss_critics = []
    recalls = []
    precisions = []
    F1s = []
    rewards = []
    valid_indices = np.arange(len(examples_valid))
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            label = torch.diag_embed(label).float()

            example = example.unsqueeze(0)
            num_nodes = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_nodes, self_loops=False)
            rel_rec, rel_send = rel_rec.float(), rel_send.float()
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()

            label_converted = torch.matmul(rel_send.t(),
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            #shape: [n_nodes, n_nodes]
            if label_converted.sum()==0:
                gID = list(range(label_converted.shape[1]))
            else:
                gID = list(get_connected_components(label_converted))
            gID = indices_to_clusters(gID)
            actions = act(example, rel_rec, rel_send)
            sims = actions_to_sims(actions, rel_rec, rel_send)
            g_predict, _ = greedy_approximate_best_clustering(sims)
            recall, precision, F1 = compute_groupMitre(gID, g_predict)
            reward = compute_reward_f1(F1)
            recalls.append(recall)
            precisions.append(precision)
            F1s.append(F1)
            rewards.append(reward)
            Q_predicted = torch.tanh(critic(example, actions, rel_rec, rel_send))
            loss = loss_critic(Q_predicted.squeeze(), torch.tensor(reward))
            loss_critics.append(loss)

    return np.mean(recalls), np.mean(precisions), np.mean(F1s), np.mean(rewards), np.mean(loss_critics)



def test():
    "test actor and critic"
    actor = torch.load(actor_file)
    critic = torch.load(critic_file)
    actor.eval()
    critic.eval()
    loss_critics = []
    recalls = []
    precisions = []
    F1s = []
    rewards = []
    test_indices = np.arange(len(examples_test))
    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            label = torch.diag_embed(label).float()

            example = example.unsqueeze(0)
            num_nodes = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_nodes, self_loops=False)
            rel_rec, rel_send = rel_rec.float(), rel_send.float()
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()

            label_converted = torch.matmul(rel_send.t(),
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            #shape: [n_nodes, n_nodes]
            #shape: [n_nodes, n_nodes]
            if label_converted.sum()==0:
                gID = list(range(label_converted.shape[1]))
            else:
                gID = list(get_connected_components(label_converted))
            gID = indices_to_clusters(gID)
            actions = torch.tanh(actor(example, rel_rec, rel_send))
            sims = actions_to_sims(actions, rel_rec, rel_send)
            g_predict, _ = greedy_approximate_best_clustering(sims)
            recall, precision, F1 = compute_groupMitre(gID, g_predict)
            reward = compute_reward_f1(F1)
            recalls.append(recall)
            precisions.append(precision)
            F1s.append(F1)
            rewards.append(reward)
            Q_predicted = torch.tanh(critic(example, actions, rel_rec, rel_send))
            loss = loss_critic(Q_predicted.squeeze(), torch.tensor(reward))
            loss_critics.append(loss)

    return np.mean(recalls), np.mean(precisions), np.mean(F1s), np.mean(rewards), np.mean(loss_critics)





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
        loss = loss_critic(Q_predicted.squeeze(), reward)
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
            
            action = torch.tanh(actor(state.float(), rel_rec.float(), rel_send.float()))
            Q = torch.tanh(critic(state.float(), action.float(), rel_rec.float(), rel_send.float()))
            #shape: [1, 1]
            actor_loss = -Q/args.batch_size
            actor_loss.backward()
            idx_count += 1

            if idx_count%args.batch_size==0:
                optimizer_actor.step()
                scheduler_actor.step()
                optimizer_actor.zero_grad()


    

def train(episode, best_F1):
    explore(episode)
    explore_good(episode)
    explore_random(episode)
    update_params(episode)
    recall_val, precision_val, F1_val, rewards_val, loss_val = validate(episode)
    print("Epoch: {:04d}".format(episode),
          "Recall: {:.10f}".format(recall_val),
          "Precision: {:.10f}".format(precision_val),
          "F1: {:.10f}".format(F1_val),
          "Loss Critic: {:.10f}".format(loss_val))
    if F1_val > best_F1:
        torch.save(actor, actor_file)
        torch.save(critic, critic_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(episode),
          "Recall: {:.10f}".format(recall_val),
          "Precision: {:.10f}".format(precision_val),
          "F1: {:.10f}".format(F1_val),
          "Loss Critic: {:.10f}".format(loss_val), file=log)
        log.flush()

    return F1_val


# Train model
t_total = time.time()
best_F1 = 0.
best_epoch = 0

for epoch in range(0, args.epochs):
    epoch += 1
    F1_val = train(epoch, best_F1)
    if F1_val > best_F1:
        best_F1 = F1_val
        best_epoch = epoch

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))

recall_test, precision_test, F1_test, rewards_test, loss_test = test()
print("Recall Test: {:.10f}".format(recall_test),
      "Precision Test: {:.10f}".format(precision_test),
       "F1 Test: {:.10f}".format(F1_test),
       "Reward Test: {:.10f}".format(rewards_test),
       "Loss Critic Test: {:.10f}".format(loss_test))

log.close()




            

            




        
        
    







    








