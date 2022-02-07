"""
Extracting data
"""

import pandas as pd
from string import ascii_lowercase, ascii_uppercase
import numpy as np

# Clean original NELA
df = pd.read_csv('data/nela-gt-2020.csv', delimiter = "\t")
df.drop(columns=['source', 'author', 'published_utc', 'collection_utc'], inplace=True)
df = df[df.label != 1]
df["label"].replace({2: 1}, inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv('data/nela-gt-2020-clean.csv', sep="\t", index=False)

# Transform to STRUCT
for key in ['title', 'text']:
    for i in ascii_lowercase:
        df[key] = df[key].str.replace(i,'a')
    for i in ascii_uppercase:
        df[key] = df[key].str.replace(i,'a')

print(df)

df.to_csv('data/nela-gt-2020_struct.csv', sep="\t", index=False)
