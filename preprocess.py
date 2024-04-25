import random
import pandas as pd
import pickle

def preprocess(data_path, rating_minimum=10, max_lines=None):
    data = pd.DataFrame(columns=['user', 'item', 'time', 'rating'])

    with open(data_path, 'r', encoding="unicode-escape") as f:
        i = 0
        for line in f:
            if line.startswith('#'):
                continue

            line = line.strip().split()

            if len(line) < 5:
                continue
            
            try:
                rating = float(line[4])
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

            data.loc[len(data)] = [user, item, int(rating_time), rating]

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

    # Map users to items
    user_to_item = {}
    items = set()
    for index, row in data.iterrows():
        user = row['user']
        item = row['item']
        items.add(item)
        if user not in user_to_item:
            user_to_item[user] = set()
        user_to_item[user].add(item)

    # Save user_to_item and items
    with open('data/user_to_item.pkl', 'wb') as f:
        pickle.dump(user_to_item, f)
    with open('data/items.pkl', 'wb') as f:
        pickle.dump(items, f)

    print('Number of users:', n_users)
    print('Number of items:', n_items)

    # Total number of interactions
    print('Number of interactions:', len(data))
    sparsity = 1 - len(data) / (n_users * n_items)
    print('Sparsity:', sparsity)

    # For each user, get last 20% of interactions
    test_data = pd.DataFrame(columns=['user', 'item', 'time', 'rating'])
    train_data = pd.DataFrame(columns=['user', 'item', 'time', 'rating'])
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        split_index = int(0.8 * len(user_data))
        train_data = pd.concat([train_data, user_data[:split_index]])
        test_data = pd.concat([test_data, user_data[split_index:]])

    # Split test_data into validation and test
    validation_data = pd.DataFrame(columns=['user', 'item', 'time', 'rating'])
    test_data = test_data.sort_values(by=['user', 'time'])
    for user in test_data['user'].unique():
        user_data = test_data[test_data['user'] == user]
        split_index = int(0.5 * len(user_data))
        validation_data = pd.concat([validation_data, user_data[:split_index]])
        test_data = test_data.drop(user_data[:split_index].index)

    print('Train data:', len(train_data))
    print('Validation data:', len(validation_data))
    print('Test data:', len(test_data))

    # Save data
    train_data.to_csv('data/train_data.csv', index=False)
    validation_data.to_csv('data/validation_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    
