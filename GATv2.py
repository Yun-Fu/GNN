from torch_geometric.nn.conv import GATv2Conv,GATConv,GCNConv,GINConv
from torch_geometric.data import Data,InMemoryDataset
from torch_geometric.loader import dataloader
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F

#123
class GATv2(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers) -> None:
        super(GATv2,self).__init__()
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels,hidden_channels))
        for i in range(num_layers-2):
            self.convs.append(GATv2Conv(hidden_channels,hidden_channels))
        self.convs.append(GATv2Conv(hidden_channels,out_channels))

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=self.convs[i](x,edge_index)
            x=F.relu(x)
            x=F.dropout(x,p=0.5,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)
#参数设置
batch_size=32    
hidden_channels=256
num_layers=2
lr=0.001
l2=0.005
epochs=300
#1.准备数据集
dataset=Planetoid('/tmp/Cora',name='Cora')
# data1=Data(x=torch.arange(0,10),edge_index=torch.randint(0,10,(2,32)),y=torch.randint(0,2,10))#一个Data类就是一个图
# dataset,slice=InMemoryDataset.collate([data1])
# data2=Data(x=torch.arange(0,5),edge_index=torch.randint(0,5,(2,10)))
# dataset,slice=InMemoryDataset.collate([data1,data2])

# train_loader=dataloader(dataset,batch_size=batch_size,shuffle=True)#因为节点分类是一个图，所以不需要批处理，直接把全图放进去了
# test_loader=dataloader(dataset,batch_size=batch_size,shuffle=False)#??测试集和训练集的划分？

#2.更据数据确定模型参数,把模型和输入都放进device
device=torch.device('cuda:1')
model=GATv2(dataset.num_features,hidden_channels,dataset.num_classes,num_layers)
model.to(device)
data=dataset[0].to(device)#data类型可以直接放进gpu
# dataset=dataset.to(device)#张量类型才能to放进gpu，dataset不能直接放进去
#3.定义优化器，损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_func=torch.nn.CrossEntropyLoss()
#4.训练
model.train()
for epoch in range(epochs):
    tot_loss=tot_acc=correct=0
    mask=data.train_mask
    optimizer.zero_grad()
    logit=model(data.x,data.edge_index)[mask]
    pred=logit.argmax(dim=-1)
    loss=loss_func(logit,data.y[mask])#计算损失函数时就是需要本身的概率值，不用argmax，只有计算正确数时才需要argmax,其次这里的损失是记录的batch个样本的平均损失，而不是一个batch的总损失，所以总损失还需要*当前batch的大小
    loss.backward()
    optimizer.step()

    tot_loss+=loss.item()#一个数的tensor用item取出值，大于一个数的张量用numpy还原成数组
    correct=float((pred==data.y[mask]).sum().item())
    tot_acc=correct/mask.sum().item()
    print(f'Epoch: {epoch}, loss: {tot_loss:.4f}, acc: {tot_acc:.4f}')

    #5.测试
    masks=[data.train_mask,data.val_mask,data.test_mask]
    accs=[]
    model.eval()
    logit=model(data.x,data.edge_index)
    pred=logit.argmax(dim=-1)
    loss=loss_func(logit,data.y).item()
    for mask in [data.train_mask,data.val_mask,data.test_mask]:
            accs.append(int((pred[mask]==data.y[mask]).sum())/int(mask.sum()))

    train_acc, val_acc, test_acc=accs
    print(f'Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')

