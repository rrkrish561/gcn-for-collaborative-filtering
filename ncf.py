from epinions_dataset import EpinionsDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF:
    """
    Neural Collaborative Filtering architecture
    """
    def __init__(self, user_input_dim, item_input_dim, embedding_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        
        self.user_embedding = nn.Linear(user_input_dim, embedding_dim)
        self.item_embedding = nn.Linear(item_input_dim, embedding_dim)

        self.lin1 = nn.Linear(embedding_dim, hidden_dim1)
        self.rel1 = nn.ReLU(hidden_dim1)
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        
    def forward(self, user, item):
        user = self.user_embedding(user)
        item = self.item_embedding(item)

        x = torch.cat([user, item], dim=1)
        x = self.lin1(x)
        x = self.rel1(x)
        x = self.lin2(x)
        return x
    