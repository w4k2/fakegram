import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB

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

quantities = np.array(
    [.0004, .0036, .0068, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
print("%i input samples" % n_samples)

all_scores = []

quantity = 0.5
scores = []
n = int(n_samples * quantity)
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
                                      ngram_range=(1, 1)).fit(corpus_b_train)

        X_train_a = extractor_a.transform(corpus_a_train).toarray()
        X_train_b = extractor_b.transform(corpus_b_train).toarray()
        X_test_a = extractor_a.transform(corpus_a_test).toarray()
        X_test_b = extractor_b.transform(corpus_b_test).toarray()

        # print(X_train_a)

        # Mierzenie sum wektorow
        vecsum_a = np.sum(X_train_a, axis=1)
        vecsum_b = np.sum(X_train_b, axis=1)
        vecsum_a_test = np.sum(X_test_a, axis=1)
        vecsum_b_test = np.sum(X_test_b, axis=1)

        stds_a = []
        stds_b = []
        means_a = []
        means_b = []

        for label in range(2):
            mask = y_train == label
            mean_a = np.mean(vecsum_a[mask])
            mean_b = np.mean(vecsum_b[mask])
            std_a = np.std(vecsum_a[mask])
            std_b = np.std(vecsum_b[mask])

            stds_a.append(std_a)
            stds_b.append(std_b)
            means_a.append(mean_a)
            means_b.append(mean_b)

            """
            print("[%i] - A = %.3f (%.2f), B = %.3f (%.2f)" % (
                label, mean_a, std_a, mean_b, std_b)
            )
            """

        # print(means_a, means_b)
        # print(stds_a, stds_b)

        gaus_a = GaussianNB().fit(vecsum_a.reshape(-1, 1), y_train)
        gaus_b = GaussianNB().fit(vecsum_b.reshape(-1, 1), y_train)

        y_pred_ga = gaus_a.predict(vecsum_a_test.reshape(-1, 1))
        y_pred_gb = gaus_b.predict(vecsum_b_test.reshape(-1, 1))
        y_pred_gc = np.argmax(np.sum(np.array([
            gaus_a.predict_proba(vecsum_a_test.reshape(-1, 1)),
            gaus_b.predict_proba(vecsum_b_test.reshape(-1, 1))
        ]),axis=0),axis=1)


        score_ga = balanced_accuracy_score(y_test, y_pred_ga)
        score_gb = balanced_accuracy_score(y_test, y_pred_gb)
        score_gc = balanced_accuracy_score(y_test, y_pred_gc)

        # print("GSCORES %.3f - %.3f" % (score_ga, score_gb))
        # exit()

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

        y_pred_z = np.argmax(np.sum(np.array([
            gaus_a.predict_proba(vecsum_a_test.reshape(-1, 1)),
            gaus_b.predict_proba(vecsum_b_test.reshape(-1, 1)),
            clf_a.predict_proba(X_test_a),
            clf_b.predict_proba(X_test_b)
        ]),axis=0),axis=1)

        # Calculate scores
        score_a = balanced_accuracy_score(y_test, y_pred_a)
        score_b = balanced_accuracy_score(y_test, y_pred_b)
        score_c = balanced_accuracy_score(y_test, y_pred_c)
        score_z = balanced_accuracy_score(y_test, y_pred_z)

        print(repeat, fold, ", ".join(["%.3f" % v for v in [score_a, score_b, score_c, score_ga, score_gb, score_gc, score_z]]))

        # exit()

        scores.append([score_a, score_b, score_c, score_ga, score_gb, score_gc])

scores = np.array(scores)

print(np.mean(scores, axis=0))
print(np.std(scores, axis=0))
