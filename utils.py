
import numpy as np
import torch 
import torch.nn as nn 



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    rel_rec = torch.from_numpy(rel_rec)
    rel_send = torch.from_numpy(rel_send)
    
    return rel_rec, rel_send