import pandas as pd
from torch.utils.data import Dataset

class EpinionsDataset(Dataset):
    def __init__(self, data_path, max_lines=200000):
        # Initialize dataframe
        self.data = pd.DataFrame(columns=['user', 'item', 'time'])


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

                # Ignore if line is not of length 5
                if len(line) < 5:
                    continue

                # If the rating is less than 3, ignore
                print(line[4])
                if float(line[4]) < 3:
                    continue
                
                user = line[1]
                item = line[0]
                rating_time = line[3]

                # Add data to dataframe
                self.data.loc[len(self.data)] = [item, user, rating_time]

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

        # Group data by user and sort by time
        self.data = self.data.sort_values(by=['user', 'time'])

        # For each user, remove latest entry and add to test set
        self.test_data = pd.DataFrame(columns=['user', 'item', 'time', 'rating'])
        for user in users:
            user_data = self.data[self.data['user'] == user]
            self.test_data = self.test_data.append(user_data.iloc[-1])
            self.data = self.data.drop(user_data.index[-1])
        
        print('Train data:', len(self.data))
        print('Test data:', len(self.test_data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]