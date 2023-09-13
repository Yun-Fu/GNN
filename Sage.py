import torch
import torch.nn.functional as F
from torch.nn import Linear,Parameter
from torch import Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops,add_remaining_self_loops
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import SAGEConv
class SAGE(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers,dropout,aggr='mean') -> None:
        super(SAGE,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels,hidden_channels,aggr=aggr))
        for i in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels,hidden_channels,aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels,out_channels,aggr=aggr))

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
aggr='mean'
#1.构建数据集
dataset=Planetoid(root='/tmp/Cora',name='Cora')
data=dataset[0]
#2.定模型
device=torch.device('cuda:1')
# model=MyGCN(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)
# model=SAGE(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout)
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
    #mean:acc: 0.8018+0.0073,max:acc: 0.7747+0.0110,lstm:
    #elu-mean:acc: 0.7971+0.0092，max：acc: 0.7792+0.0111
    model=SAGE(data.num_node_features,hidden_channels,dataset.num_classes,num_layers,dropout,aggr=aggr)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    acc=run(model)
    accs.append(acc)
accs=torch.tensor(accs)
print(f'acc: {accs.mean():.4f}+{accs.std():.4f}')
# acc: 0.8021+0.0073