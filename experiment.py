import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

df_a = pd.read_csv('data_aAa.csv')
df_b = pd.read_csv('data.csv')
corpus_a = df_a.values[:, 2].astype('U')
corpus_b = df_b.values[:, 2].astype('U')
y = df_a.values[:, -1].astype(int)

corpus_a, corpus_b, y = resample(corpus_a, corpus_b, y,
                                 random_state=1410, n_samples=1000)

corpus_a_train, corpus_a_test, corpus_b_train, corpus_b_test, y_train, y_test = train_test_split(
    corpus_a, corpus_b, y, test_size=0.25, random_state=1410)

"""
A
"""
# Extractor
extractor_a = CountVectorizer().fit(corpus_a_train)
extractor_b = CountVectorizer().fit(corpus_b_train)

X_train_a = extractor_a.transform(corpus_a_train).toarray()
X_train_b = extractor_b.transform(corpus_b_train).toarray()
X_test_a = extractor_a.transform(corpus_a_test).toarray()
X_test_b = extractor_b.transform(corpus_b_test).toarray()

# Classifier
clf_a = MLPClassifier(random_state=1410)
clf_b = MLPClassifier(random_state=1410)

print(X_train_a.shape, y_train.shape)

clf_a.fit(X_train_a, y_train)
clf_b.fit(X_train_b, y_train)

esm = np.array([clf_a.predict_proba(X_test_a),
                clf_b.predict_proba(X_test_b)])
f_esm = np.mean(esm, axis=0)

y_pred_a = clf_a.predict(X_test_a)
y_pred_b = clf_b.predict(X_test_b)
y_pred_c = np.argmax(f_esm, axis=1)

score_a = balanced_accuracy_score(y_test, y_pred_a)
score_b = balanced_accuracy_score(y_test, y_pred_b)
score_c = balanced_accuracy_score(y_test, y_pred_c)

print(score_a, score_b, score_c)
