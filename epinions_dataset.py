import torch
import pandas as pd

from torch.utils.data import Dataset

class EpinionsDataset(Dataset):
    def __init__(self, data, user_to_items, items):
        self.data = data

        self.user_to_items = user_to_items
        self.items = items

        self.num_users = len(self.user_to_items)
        self.num_items = len(self.items)
        print('Number of users:', self.num_users)
        print('Number of items:', self.num_items)

        # Map items to indices
        self.item_to_index = {item: i for i, item in enumerate(self.items)}
        # Map users to indices
        self.user_to_index = {user: i for i, user in enumerate(self.user_to_items.keys())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data.iloc[idx]['user']
        item = self.data.iloc[idx]['item']
        label = self.data.iloc[idx]['label']

        user_idx = self.user_to_index[user]
        item_idx = self.item_to_index[item]

        user_one_hot = [0] * self.num_users
        user_one_hot[user_idx] = 1

        item_one_hot = [0] * self.num_items
        item_one_hot[item_idx] = 1

        user_one_hot = torch.FloatTensor(user_one_hot)
        item_one_hot = torch.FloatTensor(item_one_hot)

        return user_one_hot, item_one_hot, label
        