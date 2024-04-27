'''
Load in the graph data and preprocess for training
'''

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

def create_mapping(path, index_col):
    data = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(data.index.unique())}
    return mapping

def load_node_csv(path, index_col, encoders=None):
    data = pd.read_csv(path, index_col=index_col)
    # mapping = {index: i for i, index in enumerate(data.index.unique())}

    data = data.reset_index()
    x = None
    if encoders is not None:
        xs = [encoder(data[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, data=None):
    if data is None:
        data = pd.read_csv(path)
    else:
        data = data

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
    
class SequenceEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


def load_graph(all_path, train_path, val_path, test_path, trust_path):
    user_mapping = create_mapping(all_path, index_col='user')
    item_mapping = create_mapping(all_path, index_col='item')

    trust_data = pd.read_csv(trust_path)

    trust_data = trust_data[trust_data['user1'].isin(user_mapping.keys()) & trust_data['user2'].isin(user_mapping.keys())]

    edge_index, _ = load_edge_csv(None, src_index_col='user1', 
                                  src_mapping=user_mapping, 
                                  dst_index_col='user2', 
                                  dst_mapping=user_mapping, 
                                  data=trust_data)

    user_data = HeteroData()
    user_data['user'].x = torch.eye(len(user_mapping))
    user_data['user1', 'trusts', 'user2'].edge_index = edge_index

    user_data = ToUndirected()(user_data)
    
    train_data = HeteroData()
    item_x = load_node_csv(all_path, index_col='item', encoders={'item': SequenceEncoder()})

    edge_index, edge_label = load_edge_csv(train_path, 
                                           src_index_col='user', 
                                           src_mapping=user_mapping,
                                           dst_index_col='item', 
                                           dst_mapping=item_mapping,
                                           encoders={'rating': IdentityEncoder(dtype=torch.long)})
    
    train_data['user'].x = torch.eye(len(user_mapping))
    train_data['item'].x = item_x
    train_data['user', 'rates', 'item'].edge_index = edge_index
    train_data['user', 'rates', 'item'].edge_label = edge_label.squeeze()

    val_data = train_data.clone()

    edge_index, edge_label = load_edge_csv(val_path, 
                                           src_index_col='user', 
                                           src_mapping=user_mapping,
                                           dst_index_col='item', 
                                           dst_mapping=item_mapping,
                                           encoders={'rating': IdentityEncoder(dtype=torch.long)})
    
    val_data['user', 'rates', 'item'].edge_label = edge_label.squeeze()
    val_data['user', 'rates', 'item'].edge_label_index = edge_index

    test_data = train_data.clone()

    edge_index, edge_label = load_edge_csv(test_path, 
                                           src_index_col='user', 
                                           src_mapping=user_mapping,
                                           dst_index_col='item', 
                                           dst_mapping=item_mapping,
                                           encoders={'rating': IdentityEncoder(dtype=torch.long)})
    
    test_data['user', 'rates', 'item'].edge_label = edge_label.squeeze()
    test_data['user', 'rates', 'item'].edge_label_index = edge_index
    test_data['user', 'rates', 'item'].edge_index = torch.cat([train_data['user', 'rates', 'item'].edge_index, val_data['user', 'rates', 'item'].edge_label_index], dim=1)

    train_data = ToUndirected()(train_data)
    val_data = ToUndirected()(val_data)
    test_data = ToUndirected()(test_data)
    del train_data['item', 'rev_rates', 'user'].edge_label # Remove "reverse" label. 
    del train_data['item', 'rev_rates', 'user'].edge_label # Remove "reverse" label. 
    del train_data['item', 'rev_rates', 'user'].edge_label # Remove "reverse" label. 

    train_data['user', 'rates', 'item'].edge_label_index = train_data['user', 'rates', 'item'].edge_index

    return train_data, val_data, test_data, user_data


if __name__ == '__main__':
    train_path = 'data/train_data.csv'
    val_path = 'data/validation_data.csv'
    test_path = 'data/test_data.csv'
    all_path = 'data/all_pos_data.csv'
    trust_path = 'data/trust.csv'
  
    train_data, val_data, test_data, user_data = load_graph(all_path, train_path, val_path, test_path, trust_path)
    print(train_data)
    print(val_data)
    print(test_data)
    print(user_data)
