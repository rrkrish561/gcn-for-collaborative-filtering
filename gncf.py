'''
Model class for graph neural collaborative filtering.
'''

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import to_hetero
import copy

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = nn.Linear(hidden_channels, out_channels)
        # self.conv2 = pyg_nn.GATConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.lin1(x).relu()
        # x = self.conv2(x, edge_index)
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
        return z

class GNCF(nn.Module):

    def __init__(self, num_users, num_items, user_emb_dim, item_emb_dim, hidden_channels, out_channels, metadata):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.item_embedding = nn.Embedding(num_items, item_emb_dim)
        
        self.encoder = GNNEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = copy.copy(x_dict)
        x_dict['user'] = self.user_embedding(x_dict['user'])
        x_dict['item'] = self.item_embedding(x_dict['item'])
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
