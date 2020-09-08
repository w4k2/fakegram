import pandas as pd
from string import ascii_lowercase, ascii_uppercase

df = pd.read_csv('data.csv')

df['title'] = df['title'].str.replace('o','a')

for key in ['author', 'title', 'text']:
    for i in ascii_lowercase:
        df[key] = df[key].str.replace(i,'a')
    for i in ascii_uppercase:
        df[key] = df[key].str.replace(i,'A')

print(df)

df.to_csv('data_aAa.csv')
