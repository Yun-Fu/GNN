import torch
import torch.nn.functional as F
from torch.nn import Linear,Parameter
from torch import Tensor
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops,remove_self_loops
from torch_geometric.nn.inits import glorot,zeros
from torch_geometric.nn import MessagePassing

#2嘿嘿
class MyGATConv2(MessagePassing):
    def __init__(self,in_channels,out_channels,heads,flow='source to target',concat=True,bias=True,dropout=0,negative_slope=0.2,**kwargs):

        kwargs.setdefault('aggr', 'add')
        super(MyGATConv2,self).__init__(node_dim=0,**kwargs)#!!!!!因为多头注意力(N,H,C)所以节点维度不再为-2了，所以要重置节点维度为0

        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_src=Linear(in_channels,heads*out_channels,bias=False)#论文全部采用glorot初始化而不是torch默认的kaiming初始化
        self.lin_dst=Linear(in_channels,heads*out_channels,bias=False)
        self.attn_src=Parameter(torch.Tensor(1,heads,out_channels))
        self.attn_dst=Parameter(torch.Tensor(1,heads,out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))#!!!!torch.Tensor随机创建的张量大部分接近与0，导致大部分神经元无效，进而导致梯度几乎为0，整个无法改变！所以必须要重置参数
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#!!!重置参数很重要！！！！不然初始化大部分为0就会导致大部分神经元都无效且无法更新
        glorot(self.lin_src.weight)
        glorot(self.lin_dst.weight)
        glorot(self.attn_src)#正交初始化防止梯度爆炸或消失？为什么？它咋这么厉害？
        glorot(self.attn_dst)
        zeros(self.bias)
        
    def forward(self,x,edge_index,edge_attr=None):
        #这里其实还需要在邻接矩阵中加入自环，这样后面计算注意力时才会同时也考虑自身，不然就只会关注邻域聚合
        num_nodes=x.size(0)
        # edge_index,_=add_self_loops(edge_index,num_nodes=num_nodes)！！！要先去掉已有的自环！！，或者add_remaining_self_loop，不能直接全部整体加自环
        edge_index, edge_attr = remove_self_loops(#要先去掉已有的自环再添加n个自环，不然会重复添加已有的自环
                    edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value='mean',
            num_nodes=num_nodes)
        
        H,C=self.heads,self.out_channels
        x_src=x_dst=self.lin_src(x).view(-1,H,C)
        x=(x_src,x_dst)

        alpha_l=(x_src*self.attn_src).sum(dim=-1)#(N,H)
        alpha_r=(x_dst*self.attn_dst).sum(dim=-1)
        alpha=(alpha_l,alpha_r)#将E次计算和每条边的内积变成了N次计算和每个节点的内积，极大减小了内存消耗和计算重复度，后面直接通过节点索引后再相加即可


        out=self.propagate(edge_index,x=x,alpha=alpha)#除了edge_index外的其它参数都必须写成键值对的形式x=x

        if self.concat:
            out=out.view(-1,H*C)#这里没有用激活函数激活，而是在外面用激活函数，这里只定义了卷积过程
        else:#mean
            out=torch.mean(out,dim=1)

        if self.bias is not None:
            out=out+self.bias

        return out
    
    def message(self,x_j,index,alpha_i,alpha_j,ptr,size_i):#求出每个源节点的初始信息表示*对应的权重得到最终的信息
        alpha=alpha_j+alpha_i#(E,H)
        alpha=F.leaky_relu(alpha,negative_slope=self.negative_slope)#默认负斜率是0.01,这里改成了论文的0.2
        alpha=softmax(alpha,index,ptr,size_i)#因为是target节点关注source节点流入的重要性，所以有多少个target节点就有多少行,dim_size=size_i
        alpha=F.dropout(alpha,p=self.dropout,training=self.training)#！！！归一化后再随机丢弃注意力权重，相当于采样邻居进行聚合
        alpha=alpha.unsqueeze(-1)#(E,H,1)之所以末尾增加一个维度是为了让同一条边的不同H计算除掉注意力权重都可以和这条边的源节点表示相乘，即(E,H,1)*(E,C)=(E,H,C)，不能直接用(E,H)*(E,C)
        return alpha*x_j
    
class MyGAT2(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_heads,num_layers,dropout=0.5) -> None:
        super(MyGAT2,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(MyGATConv2(in_channels,hidden_channels,num_heads,dropout=dropout))
        for i in range(num_layers-2):
            self.convs.append(MyGATConv2(hidden_channels*num_heads,hidden_channels,num_heads,dropout=dropout))
        self.convs.append(MyGATConv2(hidden_channels*num_heads,out_channels,1,dropout=dropout))#最后一层只能用一个头了，不然后面

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.convs[i](x,edge_index) 
            # x=F.relu(x)
            x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)

class MyGATConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels,heads,flow='source to target',concat=True,bias=True,dropout=0,negative_slope=0.2):
        super(MyGATConv,self).__init__()
        self.flow=flow
        self.heads=heads
        self.concat=concat
        self.bias=bias
        self.dropout=dropout
        self.negative_slope=negative_slope
        self.out_channels=out_channels
        self.lin_src=Linear(in_channels,heads*out_channels,bias=False)#论文全部采用glorot初始化而不是torch默认的kaiming初始化
        self.lin_dst=Linear(in_channels,heads*out_channels,bias=False)
        self.attn_src=Parameter(torch.Tensor(1,heads,out_channels))
        self.attn_dst=Parameter(torch.Tensor(1,heads,out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))#!!!!torch.Tensor随机创建的张量大部分接近与0，导致大部分神经元无效，进而导致梯度几乎为0，整个无法改变！所以必须要重置参数
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#!!!重置参数很重要！！！！不然初始化大部分为0就会导致大部分神经元都无效且无法更新
        glorot(self.lin_src.weight)
        glorot(self.lin_dst.weight)
        glorot(self.attn_src)#正交初始化防止梯度爆炸或消失？为什么？它咋这么厉害？
        glorot(self.attn_dst)
        zeros(self.bias)
        
    def forward(self,x,edge_index):
        #这里其实还需要在邻接矩阵中加入自环，这样后面计算注意力时才会同时也考虑自身，不然就只会关注邻域聚合
        edge_index,_=add_self_loops(edge_index)
        H,C=self.heads,self.out_channels
        if isinstance(x,Tensor):
            x_src=x_dst=self.lin_src(x).view(-1,H,C)
        elif isinstance(x,(tuple,list)):#说明不共用同一个特征张量
            x_src,x_dst=x
            x_src=self.lin_src(x_src).view(-1,H,C)
            x_dst=self.lin_dst(x_dst).view(-1,H,C)
        x=(x_src,x_dst)

        alpha_l=(x_src*self.attn_src).sum(dim=-1)#(N,H)
        alpha_r=(x_dst*self.attn_dst).sum(dim=-1)
        alpha=(alpha_l,alpha_r)#将E次计算和每条边的内积变成了N次计算和每个节点的内积，极大减小了内存消耗和计算重复度，后面直接通过节点索引后再相加即可

        out=self.propagate(edge_index,x,alpha)#(N,H,C)
        if self.concat:
            out=out.view(-1,H*C)#这里没有用激活函数激活，而是在外面用激活函数，这里只定义了卷积过程
        else:#mean
            out=torch.mean(out,dim=1)

        if self.bias is not None:
            out=out+self.bias

        return out 

    def propagate(self,edge_index,x,alpha):
        #确定信息流向是s->t
        j,i=(0,1) if self.flow=='source to target' else (1,0)
        edge_index_j=edge_index[j]
        edge_index_i=edge_index[i]
        #在propgate函数里完成参数数据构造！遍历1message，aggregate，update等函数的参数，如果由后缀带有_j,_i的就通过j，i索引分别为他们构造对应的表示
        #1.确定信息源节点特征,构造信息
        if isinstance(x,Tensor):
            x_j=x[edge_index_j]
        elif isinstance(x,(tuple,list)):
            x_j=x[j][edge_index_j]
        #2.根据注意力权重聚合信息
        if isinstance(alpha,Tensor):#虽然alpha肯定是一个tuple，但为了统一处理都要写出相同的形式
            alpha_j=alpha[edge_index_j]
            alpha_i=alpha[edge_index_i]
        elif isinstance(alpha,(tuple,list)):
            alpha_j=alpha[j][edge_index_j]
            alpha_i=alpha[i][edge_index_i]
        index=edge_index_i#index就是要聚合的目标节点下标
        m=self.message(x_j,index,alpha_i,alpha_j)
        a=self.aggerate(m,index)
        h=self.update(a)
        return h

    def message(self,x_j,index,alpha_i,alpha_j):#求出每个源节点的初始信息表示*对应的权重得到最终的信息
        alpha=alpha_i+alpha_j#(E,H)
        alpha=F.leaky_relu(alpha,negative_slope=self.negative_slope)#默认负斜率是0.01,这里改成了论文的0.2
        alpha=softmax(alpha,index)#因为是target节点关注source节点流入的重要性，所以有多少个target节点就有多少行,dim_size=size_i
        alpha=F.dropout(alpha,p=self.dropout,training=self.training)#！！！归一化后再随机丢弃注意力权重，相当于采样邻居进行聚合
        alpha=alpha.unsqueeze(-1)#(E,H,1)之所以末尾增加一个维度是为了让同一条边的不同H计算除掉注意力权重都可以和这条边的源节点表示相乘，即(E,H,1)*(E,C)=(E,H,C)，不能直接用(E,H)*(E,C)
        return x_j*alpha
    
    def aggerate(self,input,index):#input和index是必要参数
        return scatter(input,index,dim=0)#默认归约是求和,(E,H,C)->(N,H,C)
    
    def update(self,input):
        return input


class MyGAT(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_heads,num_layers,dropout=0.5) -> None:
        super(MyGAT,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(MyGATConv(in_channels,hidden_channels,num_heads,dropout=dropout))#没有加dropout,会过早拟合达到val最高，但实际应该向后推
        for i in range(num_layers-2):
            self.convs.append(MyGATConv(hidden_channels*num_heads,hidden_channels,num_heads,dropout=dropout))
        self.convs.append(MyGATConv(hidden_channels*num_heads,out_channels,1,dropout=dropout))#最后一层只能用一个头了，不然后面

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.convs[i](x,edge_index)
            # x=F.relu(x)
            x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)

from torch_geometric.nn.conv import GATConv
class GAT(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_heads,num_layers,dropout=0.5) -> None:
        super(GAT,self).__init__()
        self.dropout=dropout
        self.num_layers=num_layers
        self.convs=torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels,hidden_channels,num_heads,dropout=dropout))#加入自环后会改变边的权重？
        for i in range(num_layers-2):
            self.convs.append(GATConv(hidden_channels*num_heads,hidden_channels,num_heads,dropout=dropout))
        self.convs.append(GATConv(hidden_channels*num_heads,out_channels,1,dropout=dropout))#最后一层只能用一个头了，不然后面个和类别数对不上，但也可以用mean

    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=F.dropout(x,p=self.dropout,training=self.training)#！！输入卷积前先dropout，不是卷积后再dropout
            x=self.convs[i](x,edge_index)
            x=F.relu(x)
            # x=F.elu(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.convs[self.num_layers-1](x,edge_index)
        return  F.softmax(x,dim=-1)
    
from torch_geometric.datasets import Planetoid
# 定参数
hidden_channels=8
heads=8
num_layers=2
dropout=0.6
lr=0.01
l2=5e-4
epochs=300
#1.构建数据集
dataset=Planetoid(root='/tmp/Cora',name='Cora')
data=dataset[0]
print(data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
exit()
#2.定模型
device=torch.device('cuda:1')
# model=MyGAT(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout)
# model=GAT(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout)
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
        print(f'Epoch: {epoch}, train: {train_acc:.4f}, val: {val_acc:.4f}, '
          f'test: {test_acc:.4f}')
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
    
n=1
accs=[]
for i in range(n):
    print(f'----------------- {i} train -----------------')
    # model=MyGAT(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout)#acc: 0.7893+0.0128，acc: 0.7946+0.0121(dropout在卷积之前，之前在之后),acc: 0.8082+0.0104(conv内加入了卷积，丢弃注意力权重)
    # model=MyGAT2(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout)#acc: 0.7967+0.0123(没有dropout很快20以内就拟合了)，acc: 0.8097+0.0117
    model=GAT(data.num_node_features,hidden_channels,dataset.num_classes,heads,num_layers,dropout=dropout)#acc: 0.8027+0.0107，acc: 0.8106+0.0110
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    acc=run(model)
    accs.append(acc)
accs=torch.tensor(accs)
print(f'acc: {accs.mean():.4f}+{accs.std():.4f}')

# acc: 0.8115+0.0119,acc: 0.8120+0.0112
# acc: 0.8096+0.0116,0.8075+0.0107
# elu:acc: 0.8075+0.0122,acc: 0.8091+0.0118(4*16)
# relu:acc: 0.7901+0.0135(p=0，没有dropout，验证集在前20个epoch很快就达到最优了，后续训练没用了，所以要加dropout，使得其尽可能泛化性更好),
# acc: 0.8107+0.0105(p=0.6)，acc: 0.8119+0.0116,acc: 0.8140+0.0101(8*16),acc: 0.8125+0.0105(8*32),acc: 0.8055+0.0109(4*16,p=0.5)，acc: 0.8098+0.0131(4*16,p=0.7)

# best_valid=0
# best_epoch=0
# test_acc=0#在验证集上准确率最大的那次模型作为最终预测测试集的结果
# for epoch in range(epochs):
#     model.train()
#     mask=data.train_mask
#     tot_loss=tot_acc=correct=0

#     optimizer.zero_grad()
#     logits=model(data.x,data.edge_index)[mask]
#     pred=logits.argmax(dim=-1)
#     y=data.y[mask]
#     loss=loss_func(logits,y)
#     loss.backward()
#     optimizer.step()#!!!

#     tot_loss+=loss.item()
#     correct=float((pred==y).sum())
#     tot_acc=correct/mask.sum().item()
    
#     print(f'Epoch: {epoch}, loss: {tot_loss:.4f}, acc: {tot_acc:.4f}')

#     #测试
#     model.eval()
#     accs=[]
#     masks=[data.train_mask,data.val_mask,data.test_mask]
#     for mask in masks:
#         logits=model(data.x,data.edge_index)[mask]
#         pred=logits.argmax(dim=-1)
#         y=data.y[mask]
#         accs.append(float((pred==y).sum())/mask.sum().item())
#     print(f'Epoch: {epoch}, train: {accs[0]:.4f}, val: {accs[1]:.4f}, '
#           f'test: {accs[2]:.4f}')
    
#     if best_valid<accs[1]:
#         best_valid=accs[1]
#         test_acc=accs[2]
#         best_epoch=epoch

# print(f'Epoch: {best_epoch}, best_valid: {best_valid}, test acc: {test_acc}')