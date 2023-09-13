#### Imports ####

import torch
import torch.nn as nn
import random


adj_list = [[1,2,3], [0,2,3], [0, 1, 3], [0, 1, 2], [5, 6], [4,6], [4, 5], [1, 3]]
size_vertex = len(adj_list)  # number of vertices

#### Hyperparameters ####

w  = 3            # window size
d  = 2            # embedding size
y  = 200          # walks per vertex
t  = 6            # walk length 
lr = 0.025       # learning rate

v=[0,1,2,3,4,5,6,7] #labels of available vertices


#### Random Walk ####

def RandomWalk(node,t):
    walk = [node]        # Walk starts from this node
    
    for i in range(t-1):
        node = adj_list[node][random.randint(0,len(adj_list[node])-1)]
        walk.append(node)

    return walk


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi  = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))    
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))
        
        
    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out    = torch.matmul(hidden, self.phi2)
        return out

model = Model()


def skip_gram(wvi,  w):
    for j in range(len(wvi)):
        for k in range(max(0,j-w) , min(j+w, len(wvi))):
            
            #generate one hot vector
            one_hot          = torch.zeros(size_vertex)
            one_hot[wvi[j]]  = 1
            
            out              = model(one_hot)
            loss             = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
            loss.backward()
            
            for param in model.parameters():
                param.data.sub_(lr*param.grad)
                param.grad.data.zero_()


for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi=RandomWalk(vi,t)
        skip_gram(wvi, w)


print(model.phi)