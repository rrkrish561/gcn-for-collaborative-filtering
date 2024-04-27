'''
Process trust data into csv format
'''

import pandas as pd

trust_data = pd.read_csv('data/network_trust.txt', sep=' ', 
                 header=None, names=['user1', 'trust', 'user2'], 
                 na_values=['NA'], on_bad_lines='skip')

trust_data = trust_data.drop(columns=['trust'])
trust_data = trust_data.dropna()

trust_data.to_csv('data/trust.csv', index=False)

trusted_by_data = pd.read_csv('data/network_trustedby.txt', sep=' ', 
                 header=None, names=['user1', 'trustedby', 'user2'], 
                 na_values=['NA'], on_bad_lines='skip')

trusted_by_data = trusted_by_data.drop(columns=['trustedby'])
trusted_by_data = trusted_by_data.dropna()

trusted_by_data.to_csv('data/trusted_by.csv', index=False)
