import torch
from load_graph import load_graph
import argparse
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')

def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

# Load data
train_data, val_data, test_data = load_graph('data/all_pos_data.csv', 'data/train_data.csv', 'data/validation_data.csv', 'data/test_data.csv')
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

if args.use_weighted_loss:
    weight = torch.bincount(train_data['user', 'item'].edge_label)
    weight = weight.max() / weight
    print(weight)
else:
    weight = None

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.GATConv((-1, -1), hidden_channels, add_self_loops=False)
        # self.lin1 = nn.Linear(hidden_channels, out_channels)
        self.conv2 = pyg_nn.GATConv((-1, -1), out_channels, add_self_loops=False)
        # self.lin2 = nn.Linear(out_channels, out_channels)
        # super().__init__()
        # self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # x = self.conv1(x, edge_index)
        # x = self.lin1(x).relu()
        # x = self.conv2(x, edge_index)
        # x = self.lin2(x).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return x
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['item'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)

        return z.view(-1)
    
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    
model = Model(hidden_channels=32, metadata=train_data.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'item'].edge_label_index)
    target = train_data['user', 'item'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'item'].edge_label_index)
    pred = pred.clamp(min=0.5, max=5.49)
    
    target = data['user', 'item'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    num_correct = torch.sum(torch.abs(pred - target) < 0.5).item()
    return float(rmse), num_correct / len(target)

losses = []
train_losses = []
val_losses = []
test_losses = []
train_accs = [] 
val_accs = []
test_accs = []
for epoch in range(1, 301):
    loss = train()
    train_rmse, train_acc = test(train_data)
    val_rmse, val_acc = test(val_data)
    test_rmse, test_acc = test(test_data)

    losses.append(loss)
    train_losses.append(train_rmse)
    val_losses.append(val_rmse)
    test_losses.append(test_rmse)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
            f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
        
        print(f'Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')

print(f'Minimum Test RMSE: {min(test_losses):.4f}')

plt.figure()
plt.plot(losses, label='loss')
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.plot(test_losses, label='test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('runs_rating/loss.png')

plt.figure()
plt.plot(train_accs, label='train')
plt.plot(val_accs, label='val')
plt.plot(test_accs, label='test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('runs_rating/accuracy.png')

    
