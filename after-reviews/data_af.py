import pandas as pd
from string import ascii_lowercase, ascii_uppercase

df = pd.read_csv('data/data_with_labels.csv')

for key in ['title', 'article']:
    for i in ascii_lowercase:
        df[key] = df[key].str.replace(i,'a')
    for i in ascii_uppercase:
        df[key] = df[key].str.replace(i,'a')

df.to_csv('data/data_with_labels_struct.csv', index = False)
