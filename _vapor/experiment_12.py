import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

df_a = pd.read_csv('data_aAa.csv')
df_b = pd.read_csv('data.csv')
_corpus_a = df_a.values[:, 2].astype('U')
_corpus_b = df_b.values[:, 2].astype('U')
_y = df_a.values[:, -1].astype(int)

n_samples = _y.shape[0]
n_splits = 2
n_repeats = 5

# quantities = np.linspace(.01, 1, 20)
# quantities = np.array([.01, .02, .05, .1, .25, .50, .75, 1.])
quantities = np.array([.0004, .0036, .0068, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
print("%i input samples" % n_samples)

all_scores = []
for quantity in quantities:
    scores = []
    n = (n_samples * quantity).astype(int)
    print("---\n%.2f subset [%i samples]" % (quantity, n))
    for repeat in range(n_repeats):
        corpus_a, corpus_b, y = resample(_corpus_a, _corpus_b, _y,
                                         random_state=1410 + repeat,
                                         n_samples=n, stratify=_y,
                                         replace=False)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1410)

        for fold, (train, test) in enumerate(skf.split(corpus_a, y)):
            corpus_a_train = corpus_a[train]
            corpus_b_train = corpus_b[train]
            corpus_a_test = corpus_a[test]
            corpus_b_test = corpus_b[test]
            y_train = y[train]
            y_test = y[test]

            # Extractor
            extractor_a = CountVectorizer(max_features=100,
                                          ngram_range=(1, 2)).fit(corpus_a_train)
            extractor_b = CountVectorizer(max_features=100,
                                          ngram_range=(1, 2)).fit(corpus_b_train)

            X_train_a = extractor_a.transform(corpus_a_train).toarray()
            X_train_b = extractor_b.transform(corpus_b_train).toarray()
            X_test_a = extractor_a.transform(corpus_a_test).toarray()
            X_test_b = extractor_b.transform(corpus_b_test).toarray()

            # Build classifiers
            clf_a = MLPClassifier(random_state=1410)
            clf_b = MLPClassifier(random_state=1410)

            clf_a.fit(X_train_a, y_train)
            clf_b.fit(X_train_b, y_train)

            # Make ensemble
            esm = np.array([clf_a.predict_proba(X_test_a),
                            clf_b.predict_proba(X_test_b)])
            f_esm = np.mean(esm, axis=0)

            # Establish predictions
            y_pred_a = clf_a.predict(X_test_a)
            y_pred_b = clf_b.predict(X_test_b)
            y_pred_c = np.argmax(f_esm, axis=1)

            # Calculate scores
            score_a = balanced_accuracy_score(y_test, y_pred_a)
            score_b = balanced_accuracy_score(y_test, y_pred_b)
            score_c = balanced_accuracy_score(y_test, y_pred_c)

            print(repeat, fold, score_a, score_b, score_c)

            scores.append([score_a, score_b, score_c])

    scores = np.array(scores)

    print(np.mean(scores, axis=0))
    print(np.std(scores, axis=0))

    all_scores.append(scores)

all_scores = np.array(all_scores)
print(all_scores, all_scores.shape)
np.save("e12_2", all_scores)
