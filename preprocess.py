# Preproccess the epinions dataset, save the data as test and train

import pandas as pd

def preprocess(data_path, max_lines=None):
    data = pd.DataFrame(columns=['user', 'item', 'time', 'label'])

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

            data.loc[len(data)] = [user, item, rating_time, 1]

            i += 1
            if i % 10000 == 0:
                print('Processed', i, 'lines')
            if max_lines and i >= max_lines:
                break

    f.close()

    print(data.head())
    print(len(data))


    # Remove users with less than 10 ratings
    user_counts = data['user'].value_counts()
    data = data[data['user'].isin(user_counts[user_counts >= 5].index)]

    data = data.sort_values(by=['user', 'time'])
    print(data.head(100))
    
    # Iterate through dataset, remove last entry for each user and add to test data
    test_data = pd.DataFrame(columns=['user', 'item', 'time', 'label'])
    for i in range(len(data)):
        user = data.loc[i]['user']
        if i == len(data) - 1 or data.loc[i + 1]['user'] != user:
            test_data.loc[len(test_data)] = data.loc[i]
            data = data.drop(data.index[i])

    print('Train data:', len(data))
    print('Test data:', len(test_data))

    # Save the data
    data.to_csv('epinions_train.csv', index=False)
    test_data.to_csv('epinions_test.csv', index=False)