import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv,SAGEConv,GINConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch.nn import Linear,Sequential,ReLU

dataset = 'Cora'
dataset = Planetoid('/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())#包括数据集的下载，若root路径存在数据集则直接加载数据集
data = dataset[0] #该数据集只有一个图len(dataset)：1，在这里才调用transform函数
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
print(data)
# print(data.train_pos_edge_index)
# exit()
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(in_channels, 64)#Epoch: 300, Loss: 0.4114, Val: 0.9172, Test: 0.8921
        # self.conv2 = GCNConv(64, out_channels)#acc: 0.9125+0.0050

        # self.conv1 = GATConv(in_channels,8,8,dropout=0.6)#acc: 0.9063+0.0037
        # self.conv2 = GATConv(64,out_channels,1,dropout=0.6)#acc: 0.9177+0.0027

        # self.conv1 = GATv2Conv(in_channels,8,8,dropout=0.)#acc: 0.9248+0.0054
        # self.conv2 = GATv2Conv(64,out_channels,1,dropout=0.)#acc: 0.8939+0.0043

        # self.conv1 = SAGEConv(in_channels, 64,aggr='mean')#
        # self.conv2 = SAGEConv(64, out_channels,aggr='mean')#acc: 0.9024+0.0046

        # self.conv1 = SAGEConv(in_channels, 64,aggr='max')#
        # self.conv2 = SAGEConv(64, out_channels,aggr='max')#acc: 0.8911+0.0105
        
        # self.conv1=GINConv(Sequential(Linear(in_channels,64),ReLU(),Linear(64,64)))
        # self.conv2=GINConv(Sequential(Linear(64,64),ReLU(),Linear(64,out_channels)))#acc: 0.8459+0.0116

        # self.conv1=GINConv(Sequential(Linear(in_channels,64),ReLU(),Linear(64,64)),train_eps=True)
        # self.conv2=GINConv(Sequential(Linear(64,64),ReLU(),Linear(64,out_channels)),train_eps=True)#acc: 0.8580+0.0130
        

    def encode(self, x, edge_index):
        # x=F.dropout(x,p=0.6,training=self.training)#！！输入卷积前先dropout，不是卷积后再dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x=F.dropout(x,p=0.6,training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2,E]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # *：element-wise乘法

    def decode_all(self, z):
        prob_adj = z @ z.t()  # @：矩阵乘法，自动执行适合的矩阵乘法函数
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, pos_edge_index, neg_edge_index):
        return self.decode(self.encode(x, pos_edge_index), pos_edge_index, neg_edge_index)

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(data, model, optimizer, criterion):
    model.train()
    #每个epoch都重新采样一次
    neg_edge_index = negative_sampling(  # 训练集负采样，每个epoch负采样样本可能不同
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()
    # link_logits = model(data.x, data.train_pos_edge_index, neg_edge_index)
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)  # 训练集中正样本标签
    loss = criterion(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def mytest(data,model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()#计算链路存在的概率
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results
runs=10
accs=[]
for i in range(runs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, 64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)#weight_decay=5e-4,为什么加入很小的l2规范化准确率会下降这么快？
    criterion = F.binary_cross_entropy_with_logits
    best_val_auc = test_auc = best_epoch=0
    epochs=300+1
    for epoch in range(1,epochs):
        loss=train(data,model,optimizer,criterion)
        val_auc,tmp_test_auc=mytest(data,model)
        if val_auc>best_val_auc:
            best_val_auc=val_auc
            test_auc=tmp_test_auc
            best_epoch=epoch
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {tmp_test_auc:.4f}')
    print(f'Epoch: {best_epoch:03d}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')
    accs.append(test_auc)
    # #预测
    # z=model.encode(data.x,data.train_pos_edge_index)
    # final_edge_index=model.decode_all(z)
accs=torch.tensor(accs)
print(f'acc: {accs.mean():.4f}+{accs.std():.4f}')
