import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE

dataset = Planetoid("../data/cite", "CiteSeer", transform=T.NormalizeFeatures())
print(dataset.data)
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
print(data)
data = train_test_split_edges(data)
print(data)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, out_c * 2, cached=True)
        self.conv2 = GCNConv(out_c * 2, out_c, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def train():
    model.train()
    optimizer.zero_grad()
    # model.encode 调用了我们传入的编码器
    z = model.encode(x, train_pos_edge_index)
    # recon_loss 为重构损失
    loss = model.recon_loss(z, train_pos_edge_index)
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    # 使用正边和负边来测试模型的准确率
    return model.test(z, pos_edge_index, neg_edge_index)


# parameters
out_channels = 2
num_features = dataset.num_features
epochs = 100

# model
model = GAE(GCNEncoder(num_features, out_channels))
print(model)

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, epochs + 1):
    loss = train()

    # auc 指的是ROC曲线下的面积, ap 指的是平均准确度
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

# 我们可以查看模型的图嵌入
Z = model.encode(x, train_pos_edge_index)
print("\n", Z)
