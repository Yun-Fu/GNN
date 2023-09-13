import torch
import torch.nn.functional as F
from torch.nn import Linear,Parameter
from torch import Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.nn import MessagePassing

class MyGATv2Conv(MessagePassing):
    def __init__(self,in_channels,out_channels,heads,flow='source to target',concat=True,bias=True,dropout=0,negative_slope=0.2):
        super(MyGATv2Conv,self).__init__(node_dim=0)#!!!!!因为多头注意力(N,H,C)所以节点维度不再为-2了，所以要重置节点维度为0
        self.flow=flow
        self.heads=heads
        self.concat=concat
        self.bias=bias
        self.dropout=dropout
        self.negative_slope=negative_slope
        self.out_channels=out_channels
        self.lin_src=Linear(in_channels,heads*out_channels,bias=False)
        self.lin_dst=Linear(in_channels,heads*out_channels,bias=False)
        # self.attn_src=Parameter(torch.Tensor(1,heads,out_channels))
        # self.attn_dst=Parameter(torch.Tensor(1,heads,out_channels))
        self.attn=Parameter(torch.Tensor(1,heads,out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#!!!重置参数很重要！！！！不然初始化大部分为0就会导致大部分神经元都无效且无法更新
        glorot(self.lin_src.weight)
        glorot(self.lin_dst.weight)
        # glorot(self.attn_src)
        # glorot(self.attn_dst)
        glorot(self.attn)
        zeros(self.bias)
        
    def forward(self,x,edge_index):
        #这里其实还需要在邻接矩阵中加入自环，这样后面计算注意力时才会同时也考虑自身，不然就只会关注邻域聚合
        edge_index,_=add_self_loops(edge_index)
        H,C=self.heads,self.out_channels
        self.node_dim=0#!!!!!!注意pyg把节点维度默认为-2，所以选择节点是默认会从倒数第二个维度进行选择，但这里的倒数第二维度实际上是H，所以选择会出错,所以要在初始化时传入node_dim=0或人为重置node_dim=0
        tx_src=self.lin_src(x).view(-1,H,C)
        tx_dst=self.lin_dst(x).view(-1,H,C)
        # tx_src=F.leaky_relu(tx_src,negative_slope=0.2)#!!!!不能拆开分别激活，实际是对二者的和进行激活，单独对二者激活再相加就失去了原本二者之间的联系，所以必须单独算x_j和x_i的和了，没办法节省空间了
        # tx_dst=F.leaky_relu(tx_dst,negative_slope=0.2)
        # alpha_src=(self.attn_src*tx_src).sum(dim=-1)
        # alpha_dst=(self.attn_dst*tx_dst).sum(dim=-1)
        tx=(tx_src,tx_dst)
        out=self.propagate(edge_index,tx=tx)#(N,H,C)

        if self.concat:
            out=out.view(-1,H*C)#这里没有用激活函数激活，而是在外面用激活函数，这里只定义了卷积过程
        else:#mean
            out=torch.mean(out,dim=1)

        if self.bias is not None:
            out=out+self.bias

        return out 

    # def message(self,tx_j,alpha_j,alpha_i,index):
    #     a=alpha_j+alpha_i#(E,H,C)->sum->(E,H)
    #     a=softmax(a,index,dim=0)
    #     return tx_j*a.unsqueeze(-1)
    
    def message(self,tx_j,tx_i,index):
        h=tx_j+tx_i#！！！！不是cat([tx_j,tx_i],dim=-1)拼接而是+
        h=F.leaky_relu(h,negative_slope=self.negative_slope)
        a=(self.attn*h).sum(dim=-1)#(E,H,C)->sum->(E,H)
        a=softmax(a,index,dim=0)#index表示目标信息源的节点下标，s->t时是edge_index[1]
        a=F.dropout(a,p=self.dropout,training=self.training)#!!!记得加上training，不然会一直随机丢弃？
        return tx_j*a.unsqueeze(-1)


class MyGATv2(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_heads,num_layers,dropout=0.5) -> None:
        super(MyGATv2,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(MyGATv2Conv(in_channels,hidden_channels,num_heads,dropout=dropout))
        for i in range(num_layers-2):
            self.convs.append(MyGATv2Conv(hidden_channels*num_heads,hidden_channels,num_heads,dropout=dropout))
        self.convs.append(MyGATv2Conv(hidden_channels*num_heads,out_channels,1,dropout=dropout))#最后一层只能用一个头了，不然后面

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.convs[i](x,edge_index)
            # x=F.relu(x)
            x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)

from torch_geometric.nn.conv import GATv2Conv
class GATv2(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_heads,num_layers,dropout=0.5) -> None:
        super(GATv2,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels,hidden_channels,num_heads,dropout=dropout))
        for i in range(num_layers-2):
            self.convs.append(GATv2Conv(hidden_channels*num_heads,hidden_channels,num_heads,dropout=dropout))
        self.convs.append(GATv2Conv(hidden_channels*num_heads,out_channels,1,dropout=dropout))#最后一层只能用一个头了，不然后面个和类别数对不上，但也可以用mean

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)#！！输入卷积前先dropout，不是卷积后再dropout
            x=self.convs[i](x,edge_index)
            # x=F.relu(x)
            x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)
    
from torch_geometric.datasets import Planetoid
# 定参数
hidden_channels=8
heads=8
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
    model=MyGATv2(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout)#acc: 0.7918+0.0113(没有dropout)
    # model=GATv2(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout=dropout)#acc: 0.8076+0.0124
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    acc=run(model)
    accs.append(acc)
accs=torch.tensor(accs)
print(f'acc: {accs.mean():.4f}+{accs.std():.4f}')

