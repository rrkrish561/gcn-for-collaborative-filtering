import pandas as pd
from torch.utils.data import Dataset

class EpinionsDataset(Dataset):
    def __init__(self, data_path, max_lines=200000):
        # Initialize dataframe
        self.data = pd.DataFrame(columns=['user', 'item', 'rating'])


        items = set()
        users = set()

        users_to_items = {}

        # Limit to 50000 lines for now
        i = 0

        with open(data_path, 'r', encoding="unicode-escape") as f:
            for line in f:
                # Ignore first line
                if line.startswith('#'):
                    continue

                line = line.strip().split()
                
                user = line[1]
                item = line[0]
                
                # Ignore if line is not of length 5
                if len(line) < 5:
                    continue

                # Add data to dataframe
                self.data.loc[len(self.data)] = [item, user, line[4]]

                # Add user and item to set
                items.add(item)
                users.add(user)

                # Add user to item mapping
                if user in users_to_items:
                    users_to_items[user].add(item)
                else:
                    users_to_items[user] = set([item])

                i += 1
                if i % 10000 == 0:
                    print('Processed', i, 'lines')
                if i >= max_lines:
                    break



        # close file
        f.close()
                
        print(self.data.head())
        print(len(self.data))

        print('Number of users:', len(users))
        print('Number of items:', len(items))

        # Make one-hot encodings for users and items
        self.user_encodings = pd.get_dummies(self.data['user'])
        self.item_encodings = pd.get_dummies(self.data['item'])

        self.dataset = pd.DataFrame(columns=['user', 'item', 'is_rated'])

        # For each user,  item pair, add 1 if user has rated item, 0 otherwise
        print("Total num of pairs:", len(users) * len(items))
        
        for user in users:
            for item in items:
                if item in users_to_items[user]:
                    self.dataset.loc[len(self.dataset)] = [user, item, 1]
                else:
                    self.dataset.loc[len(self.dataset)] = [user, item, 0]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]