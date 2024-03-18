from epinions_dataset import EpinionsDataset
import torch.nn as nn
import torch.nn.functional as F

class NCF:
    """
    Neural Collaborative Filtering architecture
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, hidden_dim1)
        self.rel1 = nn.ReLU(hidden_dim1)
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.rel1(x)
        x = self.lin2(x)
        return x
    