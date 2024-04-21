'''
Model class for graph neural collaborative filtering.
'''

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GNCF(nn.Module):

    def __init__(self, num_users, num_items, emb_dim, hidden_channels, out_channels):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # Message is of form attention based aggregation -> linear transformation -> ReLU
        self.convUser = pyg_nn.GATConv((-1,-1), hidden_channels, heads=1)
        self.linUser = nn.Linear(hidden_channels, out_channels)

        self.convItem = pyg_nn.GATConv((-1,-1), hidden_channels, heads=1)
        self.linItem = nn.Linear(hidden_channels, out_channels)

        self.lin1 = nn.Linear(2*out_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        user_x = self.user_embedding(x['user'])
        item_x = self.item_embedding(x['item'])

        user_x = self.convUser(user_x, edge_index['user', 'rates', 'item'])
        user_x = self.linUser(user_x).relu()

        item_x = self.convItem(item_x, edge_index['item', 'rev_rates', 'user'])
        item_x = self.linItem(item_x).relu()

        x = torch.cat([user_x, item_x], dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.sigmoid(x)

        return x
