from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import Linear,Parameter
from torch import Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops,add_remaining_self_loops
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.nn import MessagePassing

class MyGCNConv(MessagePassing):
    def __init__(self,in_channels,out_channels,flow='source to target',bias=True,add_self_loops=True):#还要加bias？？？
        super(MyGCNConv,self).__init__()
        self.add_self_loops=add_self_loops
        self.flow=flow        
        self.lin=Linear(in_channels,out_channels,bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias)
        
    def forward(self,x,edge_index,edge_weight=None):
        x=self.lin(x)
        num_nodes=x.size(0)#节点数为什么使用的是edge_index.max()+1??而不是x.size(0)？
        if self.add_self_loops:
        #加入自环，考虑自身
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, None, num_nodes)

        out=self.propagate(edge_index,x=x)
        
        if self.bias is not None:
            out = out + self.bias

        return out
    
    def message(self,x_j):
        return x_j
    
class MyGCNConv(MessagePassing):
    def __init__(self,in_channels,out_channels,flow='source to target',bias=True,add_self_loops=True):#还要加bias？？？
        super(MyGCNConv,self).__init__()
        self.add_self_loops=add_self_loops
        self.flow=flow        
        self.lin=Linear(in_channels,out_channels,bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias)
    
    def degree(self,i,num_nodes):
        #GCN计算的是入度
        w=torch.ones_like(i)#每条边的权重初始化为1，根据
        w=w.unsqueeze(-1)
        deg=scatter(w,i,0,dim_size=num_nodes)
        return deg
        
    def forward(self,x,edge_index,edge_weight=None):
        x=self.lin(x)
        num_nodes=x.size(0)#节点数为什么使用的是edge_index.max()+1??而不是x.size(0)？
        if self.add_self_loops:
        #加入自环，考虑自身
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, None, num_nodes)
        j,i=edge_index# if self.flow=='source to target'
        #计算信息源的出度，若信息源就是边源，那么边源节点的出度就是信息源的出度?为什么GCN计算的是入度？
        deg=self.degree(i,num_nodes)
        deg_inv_sqrt=deg.pow(-0.5)#0的倒数是inf
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt==float('inf'),0)
        norm=deg_inv_sqrt#norm的size必须是二维？

        out=self.propagate(edge_index,x=x,norm=norm)#这边传入的张量在节点维度必须都是相同的总节点数，不然set_size时会报错，因此不能只计算i的norm，对于没出现的节点也要计算norm，只是入度为0罢了，但也要纳入计算！！！
        # print(out)#update之后的输出也是全0填充？？还是没有用转换后的x填充？？
        # print('out:',out.size())
        if self.bias is not None:
            out = out + self.bias

        return out
    
    def message(self,x_j,norm_j,norm_i):#求出每个源节点的初始信息表示*对应的权重得到最终的信息
        norm=norm_j*norm_i
        # print('x_j:',x_j.size())
        return x_j*norm
    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor or None = None, dim_size: int or None = None) -> Tensor:
        # aggregate: torch.Size([4, 64])
        # aggr-res: torch.Size([32, 64])#聚合的时候是直接scatter 4个源信息节点到目的节点上去，但还把没有涉及到的节点都用全0填充了对应的行，使得最终聚合后的表示和原始输入x的行数保持一致！
        # print('aggregate:',inputs.size())
        # res=super().aggregate(inputs, index, ptr, dim_size)
        # print(res)
        # print('aggr-res:',res.size())
        return super().aggregate(inputs, index, ptr, dim_size)
    def update(self,input):
        # print(input)#这里update的输出还是全0填充的没有聚合的点
        # print('update:',input.size())
        return input



class MyGCN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers,dropout) -> None:
        super(MyGCN,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(MyGCNConv(in_channels,hidden_channels))
        for i in range(num_layers-2):
            self.convs.append(MyGCNConv(hidden_channels,hidden_channels))
        self.convs.append(MyGCNConv(hidden_channels,out_channels))

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.convs[i](x,edge_index)
            x=F.relu(x)
            # x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)

from torch_geometric.nn.conv import GCNConv
class GCN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers,dropout) -> None:
        super(GCN,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels,hidden_channels))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))
        self.convs.append(GCNConv(hidden_channels,out_channels))

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.convs[i](x,edge_index)
            x=F.relu(x)
            # x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)
    
from torch_geometric.datasets import Planetoid
# 定参数
hidden_channels=64
num_layers=2
dropout=0.6
lr=0.005
l2=5e-4
epochs=100
#1.构建数据集
dataset=Planetoid(root='/tmp/Cora',name='Cora')
data=dataset[0]
#2.定模型
device=torch.device('cuda:1')
# model=MyGCN(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)
# model=GCN(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)
#注意输入的节点数和输出表示的节点数是一样的！不会因为edge_index里只包含m个节点就只聚合m个节点，还是会输出n个节点的表示?是因为加了自环的原因吧？如果不加自环呢？为什么不加自环也有n个节点表示？
# model.to(device)
data=data.to(device)
#3.定损失函数
loss_func=torch.nn.CrossEntropyLoss()
#4.定优化器
# optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)

#开训

def run(model):
    best_valid=0
    best_epoch=0
    acc=0#在验证集上准确率最大的那次模型作为最终预测测试集的结果
    for epoch in range(epochs):
        train_loss,train_acc=train(model)
        # print(f'Epoch: {epoch}, loss: {train_loss:.4f}, acc: {train_acc:.4f}')
        train_acc, val_acc, test_acc=test(model)
        # print(f'Epoch: {epoch}, train: {train_acc:.4f}, val: {val_acc:.4f}, '
        #   f'test: {test_acc:.4f}')
        if best_valid<val_acc:
            best_valid=val_acc
            acc=test_acc
            best_epoch=epoch
    print(f'Epoch: {best_epoch}, best_valid: {best_valid}, test acc: {acc}')
    return acc

def train(model):
    model.train()
    mask=data.train_mask
    tot_loss=tot_acc=correct=0

    optimizer.zero_grad()
    logits=model(data.x,data.edge_index)[mask]
    pred=logits.argmax(dim=-1)
    y=data.y[mask]
    loss=loss_func(logits,y)
    loss.backward()
    optimizer.step()#!!!

    tot_loss+=loss.item()
    correct=float((pred==y).sum())
    tot_acc=correct/mask.sum().item()
    
    return tot_loss,tot_acc
    print(f'Epoch: {epoch}, loss: {tot_loss:.4f}, acc: {tot_acc:.4f}')

def test(model):
    #测试
    model.eval()
    accs=[]
    masks=[data.train_mask,data.val_mask,data.test_mask]
    for mask in masks:
        logits=model(data.x,data.edge_index)[mask]
        pred=logits.argmax(dim=-1)
        y=data.y[mask]
        accs.append(float((pred==y).sum())/mask.sum().item())
    return accs
    print(f'Epoch: {epoch}, train: {accs[0]:.4f}, val: {accs[1]:.4f}, '
          f'test: {accs[2]:.4f}')
    
n=100
accs=[]
for i in range(n):
    print(f'----------------- {i} train -----------------')
    # model=MyGCN(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)#acc: 0.8019+0.0088,acc: 0.8128+0.0067(加入剩下的自环)
    model=GCN(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)#acc: acc: 0.8139+0.0076
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    acc=run(model)
    accs.append(acc)
accs=torch.tensor(accs)
print(f'acc: {accs.mean():.4f}+{accs.std():.4f}')
# acc: 0.7659+0.0045
