import random
import pandas as pd
import pickle

def preprocess(data_path, rating_minimum=10, max_lines=None):
    data = pd.DataFrame(columns=['user', 'item', 'time', 'label'])

    user_to_item = {}
    users = set()
    items = set()

    with open(data_path, 'r', encoding="unicode-escape") as f:
        i = 0
        for line in f:
            if line.startswith('#'):
                continue

            line = line.strip().split()

            if len(line) < 5:
                continue
            
            try:
                if float(line[4]) < 3:
                    continue
            except:
                continue
            
            item = line[0]
            user = line[1]
            rating_time = line[3]

            # Ignore if time is not an integer
            try:
                int(rating_time)
            except:
                continue

            data.loc[len(data)] = [user, item, int(rating_time), 1]
            users.add(user)
            items.add(item)
            if user not in user_to_item:
                user_to_item[user] = set()
                user_to_item[user].add(item)
            else:
                user_to_item[user].add(item)

            i += 1
            if i % 10000 == 0:
                print('Processed', i, 'lines')
            if max_lines and i >= max_lines:
                break

    f.close()
    data = data.groupby('user').filter(lambda x: len(x) >= rating_minimum)
    # Sort by user and time
    data = data.sort_values(by=['user', 'time'])
    # Reset indices
    data = data.reset_index(drop=True)

    # Count number of users and items
    n_users = len(data['user'].unique())
    n_items = len(data['item'].unique())

    print('Number of users:', n_users)
    print('Number of items:', n_items)

    # Total number of interactions
    print('Number of interactions:', len(data))
    sparsity = 1 - len(data) / (n_users * n_items)
    print('Sparsity:', sparsity)

    # For each user, get the last item rated, maintaining the original index
    test = data.groupby('user').tail(1).reset_index()
    print("Test set size:", len(test))
    print(test.head(20))

    # Remove all indices that are in the test set from the train set
    train = data[~data.index.isin(test['index'])]
    print("Number of interactions after removing test set:", len(train))

    # Generate 4 random negative samples for each positive sample in the train set
    negative_samples = pd.DataFrame(columns=['user', 'item', 'time', 'label'])
    for index, row in train.iterrows():
        user = row['user']
        item = row['item']
        time = row['time']

        neg_items = set()

        while len(neg_items) < 4:
            neg_item = random.choice(list(items-user_to_item[user]))
            neg_items.add(neg_item)
        
        for neg_item in neg_items:
            negative_samples.loc[len(negative_samples)] = [user, neg_item, time, 0]
        
    train = pd.concat([train, negative_samples])
    # Sort by user and time
    train = train.sort_values(by=['user', 'time'])
    print("Train set size after adding negative samples:", len(train))
    print(train.head(20))

    # Save test set and train set
    test.to_csv('test.csv', index=False)
    train.to_csv('train.csv', index=False)

    # pickle set of items and user to items
    with open('items.pkl', 'wb') as f:
        pickle.dump(items, f)

    with open('user_to_item.pkl', 'wb') as f:
        pickle.dump(user_to_item, f)
