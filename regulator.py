import pandas as pd
import numpy as np

df_words = pd.read_csv('data/data.csv')
df_struct = pd.read_csv('data/data_struct.csv')

print(df_words.shape)
print(df_struct.shape)

X_words = []
X_struct = []
y = df_words['label'].values.astype(int)

print(y.shape)

keys = ['author', 'title', 'text']

for key in keys:
    X_words.append(df_words[key].values.astype('U'))
    X_struct.append(df_struct[key].values.astype('U'))

X_words = np.array(X_words).T
X_struct = np.array(X_struct).T

print(X_words.shape)
print(X_struct.shape)

np.save("data/y", y)
np.save("data/X_words", X_words)
np.save("data/X_struct", X_struct)
