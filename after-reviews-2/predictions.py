import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import scipy as sp
from sklearn.base import clone

"""
Parameters
"""
keys = ['text', 'title']
i_s = ['words', 'struct']
n_range = 6
n_splits = 2
n_repeats = 5
quantities = np.array([.02, .05, .1, .25, .40])
random_state = 1410

print("# Load CSV")
df_words = pd.read_csv('data/nela-gt-2020.csv', delimiter = "\t")
y = df_words['label'].values.astype(int)

s_idx = np.array(range(len(y))).astype(int)
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)
base_clf = MLPClassifier(random_state=1410)
print(base_clf)

# Extraction loop
for key in keys:
    print("# Key %s" % key)
    for i, base in enumerate(i_s):
        print("## Base %s" % base)
        for repeat in range(n_repeats):
            print("### Repeat %i" % repeat)
            for q_id, quantity in enumerate(quantities):
                print("#### Quantity %.2f [%i]" % (quantity, q_id))
                # Quantity resampling
                resampled = resample(s_idx, n_samples=int(len(y)*quantity),
                                     replace=False, stratify=y,
                                     random_state=random_state + repeat)

                # print(resampled.shape)
                for fold, (train, test) in enumerate(skf.split(y[resampled],
                                                               y[resampled])):
                    print("##### Fold %i" % fold)
                    preds = []
                    probas = []
                    for n_start in range(n_range):
                        for n_end in range(n_range):
                            if n_start <= n_end:
                                n_ran = (n_start+1, n_end+1)

                                filename = "%i_%i_%i_%s_%s_%i_%i" % (repeat, q_id, fold, key, i_s[i], *n_ran)

                                X = np.load("extracted/%s.npy" % filename, allow_pickle=True)[()].A

                                # print(X.shape)

                                clf = clone(base_clf)
                                clf.fit(X[train], y[resampled][train])

                                y_pred = clf.predict(X[test])
                                proba = clf.predict_proba(X[test])

                                score = balanced_accuracy_score(y[resampled][test], y_pred)

                                print("%.3f | %s" % (score, filename), y_pred.shape)

                                preds.append(y_pred)
                                probas.append(proba)

                                X = None

                    preds = np.array(preds)
                    probas = np.array(probas)
                    print(preds.shape, probas.shape)
                    print(filename[:-4], "%.3f" % np.std(preds), preds.shape, probas.shape)
                    np.save("predictions/%s" % filename[:-4], preds)
                    np.save("probas/%s" % filename[:-4], probas)

                resampled = None
