import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

df_a = pd.read_csv('data_aAa.csv')
df_b = pd.read_csv('data.csv')
df_c = pd.read_csv('data_bbb.csv')
_corpus_a = df_a.values[:, 2].astype('U')
_corpus_b = df_b.values[:, 2].astype('U')
_corpus_c = df_c.values[:, 2].astype('U')
_y = df_a.values[:, -1].astype(int)

n_samples = _y.shape[0]
n_splits = 2
n_repeats = 5
n_gram_max = 7

print("%i input samples" % n_samples)

all_scores = []

quantity = .20

# FOLDS x METHOD x I x J
scores = np.zeros((n_splits * n_repeats, 3, n_gram_max, n_gram_max))
n = int(n_samples * quantity)
print("---\n%.2f subset [%i samples]" % (quantity, n))
for repeat in range(n_repeats):
    corpus_a, corpus_b, corpus_c, y = resample(_corpus_a, _corpus_b, _corpus_c, _y,
                                               random_state=1410 + repeat,
                                               n_samples=n, stratify=_y,
                                               replace=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1410)

    for fold, (train, test) in enumerate(skf.split(corpus_a, y)):
        corpus_a_train = corpus_a[train]
        corpus_b_train = corpus_b[train]
        corpus_c_train = corpus_c[train]
        corpus_a_test = corpus_a[test]
        corpus_b_test = corpus_b[test]
        corpus_c_test = corpus_c[test]
        y_train = y[train]
        y_test = y[test]

        # Extractor
        for i in range(n_gram_max):
            for j in range(n_gram_max):
                if i > j:
                    continue
                print(i + 1, j + 1)
                extractor_a = CountVectorizer(max_features=100,
                                              ngram_range=(i + 1, j + 1)).fit(corpus_a_train)
                extractor_b = CountVectorizer(max_features=100,
                                              ngram_range=(i + 1, j + 1)).fit(corpus_b_train)
                extractor_c = CountVectorizer(max_features=100,
                                              ngram_range=(i + 1, j + 1)).fit(corpus_c_train)

                X_train_a = extractor_a.transform(corpus_a_train).toarray()
                X_train_b = extractor_b.transform(corpus_b_train).toarray()
                X_train_c = extractor_c.transform(corpus_c_train).toarray()
                X_test_a = extractor_a.transform(corpus_a_test).toarray()
                X_test_b = extractor_b.transform(corpus_b_test).toarray()
                X_test_c = extractor_c.transform(corpus_c_test).toarray()

                # Build classifiers
                clf_a = MLPClassifier(random_state=1410)
                clf_b = MLPClassifier(random_state=1410)
                clf_c = MLPClassifier(random_state=1410)

                clf_a.fit(X_train_a, y_train)
                clf_b.fit(X_train_b, y_train)
                clf_c.fit(X_train_c, y_train)

                # Establish predictions
                y_pred_a = clf_a.predict(X_test_a)
                y_pred_b = clf_b.predict(X_test_b)
                y_pred_c = clf_c.predict(X_test_c)

                # Calculate scores
                score_a = balanced_accuracy_score(y_test, y_pred_a)
                score_b = balanced_accuracy_score(y_test, y_pred_b)
                score_c = balanced_accuracy_score(y_test, y_pred_c)

                print(score_a, score_b, score_c)

                scores[fold + 2 * repeat, 0, i, j] = score_a
                scores[fold + 2 * repeat, 1, i, j] = score_b
                scores[fold + 2 * repeat, 2, i, j] = score_c

                print(scores)

np.save("n_gram_scores", scores)
