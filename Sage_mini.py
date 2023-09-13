from torch_geometric.data import NeighborSampler,Data
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as  F
import time
import tqdm

device=torch.device('cuda:1')

class Sage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,num_layers) -> None:
        super(Sage,self).__init__()
        self.num_layers=num_layers

        self.convs=nn.ModuleList()
        self.convs.append(SAGEConv(in_channels,hidden_channels))
        for i in range(1,num_layers-1):
            self.convs.append(SAGEConv(hidden_channels,hidden_channels))
        self.convs.append(SAGEConv(hidden_channels,out_channels))
    
    def forward(self,x,adjs):#x是涉及到这n阶邻居的所有节点特征
        for i, (edge_index,e_id,size) in enumerate(adjs):
            x_target=x[:size[1]]
            x=self.convs[i]((x,x_target),edge_index)#！！！！！注意这里的边edge_index都不再是原始图的节点下标了，而是映射到0-n_id的节点的新下标了,因为换了图，图中节点的编号自然也要重新排序了
            #为什么输入的是一个元组？(src,target),目的区分要求的目标节点，不然会把所有的x都当做目标节点，但实际上这一层是专门聚合l-1层邻居即h(l-1)
            #因为是采样所以只有target节点才才包括已经聚合了l-1次的节点，所以w(l)才能对这些聚合了l-1次的节点进行操作
            #而实际采样只选了流向target节点的source邻居节点，而这些邻居节点并没有流进的节点，因此无法为底层的邻居聚合邻居得到新一层的节点表示
            #每层中间加入非线性激活层
            if i!=self.num_layers-1:
                x=F.relu(x)
                x=F.dropout(x,p=0.5,training=self.training)
        return x.log_softmax(dim=-1)
    
    def inference(self,x_all,subgraph_loader):
        for i in range(self.num_layers):
            xs=[]
            # 一个loader里装的是n/batch_size个batch的数据
            # 每个batch里面有batch_size个目标节点，及这些采样m层的所有m层邻接矩阵adjs
            # 所以每个batch里还要遍历adjs得到每层采样的邻接矩阵adjs[i],而如果只采样了一层，则就只有一个adj，就不用遍历adjs

            # shuffle=False并不会保证本batch的图里所涉及的所有节点n_id全部是按照顺序排列的，但能保证所有的目标节点n_id[:batch_size/size[1]]都是按照顺序排列的！
            # 而其它采样得到的邻居则没有顺序，这也很好理解，因为每次迭代其实就是随机选出batch个目标节点，然后再采样邻居，
            # 虽然采样的邻居无关顺序，但我们可以保证每次选择目标节点的顺序就是从小到大的！
            for batch_size,n_id,adj in subgraph_loader:
                edge_index,_,size=adj.to(device)#只采样了一阶邻居，所以不用遍历adjs，因为只有一个adj
                x=x_all[n_id].to(device)
                x_target=x[:size[1]]
                x=self.convs[i]((x,x_target),edge_index)
                if i!=self.num_layers-1:
                    x=F.relu(x)
                xs.append(x.cpu())
                #x表示一个batch_size个节点的表示，注意这里的n_id必须从0-n-1排列，所以不能用shuffle打乱，否则后面直接用cat堆叠起来就对应不上
                #比如若当前第一个batch的n_id不是[0,1,2]而是[6,2,8],这样计算出一个batch的节点表示因为是按顺序拼接的，所以还是会放在前三行作为[0,1,2]的最终节点节点表示！这样顺序就乱了！
            x_all=torch.cat(xs,dim=0).to(device)#简单的合并只有在shuffle=False才可以用，否则要将节点重新放在n_id对应的行上！
        return  x_all.log_softmax(dim=-1)
    
    def inference2(self,x_all,subgraph_loader):#当shuffle！=False时就只能用这个函数了，要对生成的节点表示重新放在它对应的行n_id上
        from torch_scatter import scatter
        for i in range(self.num_layers):
            index=[]
            xs=[]
            for batch_size,n_id,adj in subgraph_loader:
                edge_index,_,size=adj.to(device)#只采样了一阶邻居，所以不用遍历adjs，因为只有一个adj
                x=x_all[n_id].to(device)
                x_target=x[:size[1]]
                x=self.convs[i]((x,x_target),edge_index)
                if i!=self.num_layers-1:
                    x=F.relu(x)
                index.append(n_id[:batch_size])
                xs.append(x)#新的节点表示放在对应的行上,虽然一个epoch是全图，但一个batch里并没有包括全部节点，所以目标节点还是前size[1]个节点！
            xs=torch.cat(xs,dim=0).to(device)
            index=torch.cat(index,dim=-1).squeeze(0).to(device)
            x_all=scatter(xs,index,dim=0).to(device)#
        return  x_all.log_softmax(dim=-1)

def train_epoch(epoch,model,dataset,train_loader):#一个epoch迭代完一个数据集的所有batch
    tot_loss=tot_correct=tot_example=0
    for i,batch_data in enumerate(train_loader):
        batch_size,n_id,adjs=batch_data
        #定义当前batch的输入，输出
        x=dataset.x[n_id].to(device)#!!!!注意这里的输入只能传x[n_id]不能穿x，因为只有取了n_id的节点后面才能从0到n_id重新为节点编号
        y=dataset.y[n_id[:batch_size]].to(device)#注意n_id是一张图里面的所有节点，因此还包含了非目标节点，所以实际标签只取前面的batch_size个节点
        
        optimizer.zero_grad()
        adjs=[adj.to(device) for adj in adjs]
        y_hat=model(x,adjs)#！！！一定注意这里不能直接传dataset.x进去，要对节点进行重新编号只能先取出来
        loss=loss_func(y_hat,y)
        loss.backward()
        optimizer.step()

        tot_loss+=float(loss)*batch_size#注意最后一层可能不满设定的batch_size，所以要单独乘
        tot_correct+=int((y_hat.argmax(dim=-1)==y).sum())
        tot_example+=batch_size

    loss_epoch,acc_epoch=tot_loss/tot_example,tot_correct/tot_example
    return loss_epoch,acc_epoch
        
def test(model,dataset,subgraph_loader):
    model.eval()#先将模型设置为非梯度状态！
    x=dataset.x.to(device)#!!!!
    y_hat=model.inference2(x,subgraph_loader).argmax(dim=-1)
    y=dataset.y.to(device)#！！！！注意数据集标签必须要放进device里
    accs=[]
    for mask in [dataset.train_mask,dataset.val_mask,dataset.test_mask]:
        accs.append(int((y_hat[mask]==y[mask]).sum())/int(mask.sum()))
    return accs

#初始特征从哪里来？data.x就是下标对应的节点的特征！x不是记录的节点，而是节点的特征！
#1.确定数据集，根据数据集确定模型输入，输出，以及模型的参数大小
dataset=Planetoid(root='/tmp/Cora',name='Cora')
print(dataset.x.size())
print(dataset.num_features)
print(dataset.num_classes)
#2.定义数据集加载器loader
train_loader=NeighborSampler(dataset.edge_index,node_idx=dataset.train_mask,#!!!注意这里的node_idx并不直接是节点的id列表，而是通过mask标记的
                             sizes=[25,10],batch_size=256,shuffle=True,
                             num_workers=12)#选的邻居点是流向target节点的source节点，不是从target流出的！
#！！！注意验证的时候必须把目标节点设置为全部节点，即node_idx=None！！！且shuffle必须为false保证n_id从0-n-1排序
#因为验证的时候节点很多了且要用到所有的邻居了，如果再单独为每个节点创建一个n阶子图可能会占用很大的空间，
#所以这里直接只采样了一阶子图，然后迭代n次，但这一阶子图是要在l层聚合时重复使用的，所以必须要保证这个子图足够大，
# 而不能只采样指向目标节点的一阶邻居，如果这样的话，那么这些一阶邻居也没有办法聚合l-1层的信息，所以为了采样空间随着层数的增大而增大
# 直接采样全图的每个节点的一阶邻居，相当于只输入全图一次，而如果针对每个节点都单独采样n次，则会采样全图很多次，空间会爆
subgraph_loader=NeighborSampler(dataset.edge_index,node_idx=None,sizes=[-1],
                                batch_size=1024,shuffle=False,#就算设定成false，每个batch不随机打乱，但每个batch也还是有部分节点不是按照0-n-1的顺序生成的啊
                                num_workers=12)
#3.根据数据集特征大小定义模型参数
model=Sage(dataset.num_features,256,dataset.num_classes,num_layers=2)
model.to(device)#!!!!!!模型，数据等张量***数据都要放到gpu上去，不能一个在cpu一个在gpu
#4.定义优化器
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
#5.定义损失函数
loss_func=torch.nn.CrossEntropyLoss()
#6.开始训练
epochs=300
model.train()
for epoch in range(epochs):
    loss,acc=train_epoch(epoch,model,dataset,train_loader)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Acc: {acc}') 
    train_acc, val_acc, test_acc=test(model,dataset,subgraph_loader)
    print(f'Epoch {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

