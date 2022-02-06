"""
Extracting data
"""

import pandas as pd
from string import ascii_lowercase, ascii_uppercase

df = pd.read_csv('data/nela-gt-2020.csv', delimiter = "\t")

for key in ['title', 'text']:
    for i in ascii_lowercase:
        df[key] = df[key].str.replace(i,'a')
    for i in ascii_uppercase:
        df[key] = df[key].str.replace(i,'a')

print(df)

df.to_csv('data/nela-gt-2020_struct.csv')
