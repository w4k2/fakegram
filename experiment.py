import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

df = pd.read_csv('data_aAa.csv')
corpus = df.values[:, 2].astype('U')
y = df.values[:, -1].astype(int)

corpus, y = resample(corpus, y, random_state=1410, n_samples=1000)

corpus_train, corpus_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.25, random_state=1410)

# Extractor
extractor = CountVectorizer().fit(corpus_train)

X_train = extractor.transform(corpus_train).toarray()
X_test = extractor.transform(corpus_test).toarray()

# Classifier
clf = MLPClassifier(random_state=1410)

print(X_train.shape, y_train.shape)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred)
score = balanced_accuracy_score(y_test, y_pred)

print(score)
