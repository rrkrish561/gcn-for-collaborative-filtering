'''
Initializes the graph of the Epinions dataset.
'''

import pandas as pd
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import HeteroData
import pickle
import torch

train_data = pd.read_csv('data/train.csv')

print(train_data.head())

# Get user to items dictionary from pickle file
with open('data/user_to_item.pkl', 'rb') as f:
    user_to_items = pickle.load(f)

# Get items set from pickle file
with open('data/items.pkl', 'rb') as f:
    items_set = pickle.load(f)

def load_node_csv(path, index_col, encoders=None):
    data = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(data.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(data[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
    data = pd.read_csv(path)

    src = [src_mapping[index] for index in data[src_index_col]]
    dst = [dst_mapping[index] for index in data[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(data[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
    
# Return node features and mapping from user/item to index
_, user_mapping = load_node_csv('data/train.csv', index_col='user')
_, item_mapping = load_node_csv('data/train.csv', index_col='item')

# Return edge index and edge attributes
edge_index, edge_label = load_edge_csv('data/train.csv', 
                                      src_index_col='user', 
                                      src_mapping=user_mapping, 
                                      dst_index_col='item', 
                                      dst_mapping=item_mapping,
                                      encoders={'label': IdentityEncoder(dtype=torch.long)})
    

data = HeteroData()
data['user', 'rates', 'item'].edge_index = edge_index
data['user', 'rates', 'item'].edge_label = edge_label

# Add a reverse edge for user aggregration/item modeling
data = ToUndirected()(data)

print(data)
