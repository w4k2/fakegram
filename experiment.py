import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_aAa.csv')
corpus = df.values[:, 2]
y = df.values[:, -1]
print(corpus)
print(y)

corpus_train, corpus_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.25, random_state=1410)

# Extractor
print(corpus_train)
#extractor = CountVectorizer().fit(corpus_train)

print("Extracted")
