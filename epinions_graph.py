'''
Initializes the graph of the Epinions dataset.
'''

import pandas as pd
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import negative_sampling, structured_negative_sampling
from torch_geometric.sampler import NegativeSampling
from torch_geometric.data import HeteroData
import pickle
import torch

train_data = pd.read_csv('data/train.csv')
train_idx = len(train_data)
test_data = pd.read_csv('data/test.csv')
all_data = pd.read_csv('data/all_data.csv')

print(all_data.head())

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
_, user_mapping = load_node_csv('data/all_data.csv', index_col='user')
_, item_mapping = load_node_csv('data/all_data.csv', index_col='item')

# Return edge index and edge attributes
edge_index, edge_label = load_edge_csv('data/all_data.csv', 
                                      src_index_col='user', 
                                      src_mapping=user_mapping, 
                                      dst_index_col='item', 
                                      dst_mapping=item_mapping,
                                      encoders={'label': IdentityEncoder(dtype=torch.long)})
    

data = HeteroData()
# data['user'].x = torch.eye(len(user_mapping))
# data['item'].x = torch.eye(len(item_mapping))
data['user'].x = torch.arange(len(user_mapping))
data['item'].x = torch.arange(len(item_mapping))
data['user', 'rates', 'item'].edge_index = edge_index
data['user', 'rates', 'item'].edge_label = edge_label

train_data = data.clone()
test_data = data.clone()

train_data['user', 'rates','item'].edge_index = data['user', 'rates', 'item'].edge_index[:, :train_idx]
train_data['user', 'rates','item'].edge_label = data['user', 'rates', 'item'].edge_label[:train_idx]
train_data['user', 'rates', 'item'].edge_label_index = data['user', 'rates', 'item'].edge_index[:, :train_idx]

test_data['user', 'rates','item'].edge_index = data['user', 'rates', 'item'].edge_index[:, :train_idx]
test_data['user', 'rates','item'].edge_label = data['user', 'rates', 'item'].edge_label[train_idx:]
test_data['user', 'rates', 'item'].edge_label_index = data['user', 'rates', 'item'].edge_index[:, train_idx:]

# Add a reverse edge for user aggregration/item modeling
train_data = ToUndirected()(train_data)
del train_data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
test_data = ToUndirected()(test_data)
del test_data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.


num_users = len(user_mapping)
num_items = len(item_mapping)

# Add negtive sampling for test data
neg_edge_list = []
for user in range(num_users):
    negative_samples = set()
    user_neighbors = edge_index[1, edge_index[0] == user]
    while len(negative_samples) < 100:
        neg_item = torch.randint(0, num_items, (1,)).item()
        if neg_item not in user_neighbors:
            negative_samples.add(neg_item)

    neg_edges = torch.tensor([[user] * 100, list(negative_samples)])
    neg_edge_list.append(neg_edges)

neg_edge_label_index = torch.cat(neg_edge_list, dim=1)
test_data['user', 'rates', 'item'].neg_edge_label = torch.zeros(neg_edge_label_index.size(1)).unsqueeze(1)
test_data['user', 'rates', 'item'].neg_edge_label_index = neg_edge_label_index

print(data)
print(train_data)
print(test_data)
