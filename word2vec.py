from gensim.models.word2vec import Word2Vec,KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
import random
loss_func=nn.BCELoss()#二分类损失函数
dataset=Planetoid(root='/tmp/Cora',name='Cora')
data=dataset[0]
j,i=data.edge_index
num_nodes=data.num_nodes
adj=SparseTensor(row=j,col=i)
l=10#每个节点采样序列的长度1
n=10#每个节点采样序列的次数
w=5#窗口大小
class skip_gram(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.u=nn.Embedding()
        self.v=nn.Embedding()

    def decode(self,edge_index):
        return F.sigmoid(self.u[edge_index[0]]*self.v[edge_index[1]]).sum(dim=-1)#返回每对uv节点是否会出现在同一窗口的概率
    
def randomWalk(u,l,adj):#从节点u开始随机游走长为l的随机游走序列
    walk=[u]
    for i in range(l-1):
        neighbors=adj[u].storage.col()
        if len(neighbors)>0:
            next=neighbors[torch.randint(len(neighbors),size=(1,))].item()
            walk.append(next)
            u=next
        else:
            break
    return walk

def selectAllwalks(nodes,l,n,adj):
    walks=[]
    for c in range(n):#每个节点跑n次随机游走
        for u in nodes:
            walks.append(randomWalk(u,l,adj))
    return walks

w2v_vpath='./GNN/model/w2v.bin'
w2v_mpath='./GNN/model/w2v.model'
# walks=selectAllwalks(range(num_nodes),l,n,adj)
# model=Word2Vec(walks,sg=1,window=w,workers=5,negative=5)
# model.wv.save_word2vec_format(w2v_vpath)
# model.save(w2v_mpath)

vectors=KeyedVectors.load_word2vec_format(w2v_vpath).vectors
print(type(vectors))
vectors=torch.tensor(vectors)
e=torch.nn.Embedding.from_pretrained(vectors)
# b=e(vectors)

print(e(torch.tensor([1,2])))
# model=Word2Vec.load(w2v_mpath)
# print(model.wv.similarity(0,1))
# print(model.wv.most_similar(0))

import torch.nn as nn
class classifier(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb=nn.Embedding.from_pretrained(vectors,freeze=True)
        self.lin1=nn.Linear(in_c,)