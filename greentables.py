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
keys = ['text', 'author', 'title']
i_s = ['words', 'struct']
n_range = 6
n_splits = 2
n_repeats = 5
quantities = np.array([.02, .05, .1, .25, .40])
random_state = 1410

print("# Load CSV")
df_words = pd.read_csv('data/data.csv')
y = df_words['label'].values.astype(int)

s_idx = np.array(range(len(y))).astype(int)
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)
base_clf = MLPClassifier(random_state=1410)
print(base_clf)

# Scores
# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.zeros((len(keys), len(i_s), n_repeats, n_splits, len(quantities), n_range, n_range))
probas_array = np.zeros((len(keys), len(i_s), n_repeats, n_splits, len(quantities), n_range, n_range), dtype=object)
ytest_array = np.zeros((n_repeats, n_splits, len(quantities)), dtype=object)

print(scores.shape)


# Extraction loop
for key_id, key in enumerate(keys):
    print("# Key %s" % key)
    for i, base in enumerate(i_s):
        print("## Base %s" % base)
        for repeat in range(n_repeats):
            print("### Repeat %i" % repeat)
            for q_id, quantity in enumerate(quantities):
                print("#### Quantity %.2f [%i]" % (quantity, q_id))
                # Quantity resampling
                filename = "%i_%i_%s_%s" % (repeat, q_id, key, i_s[i])
                resampled = np.load("cozio/res_%s.npy" % filename)

                # print(resampled.shape)
                for fold, (train, test) in enumerate(skf.split(y[resampled],
                                                               y[resampled])):
                    print("##### Fold %i" % fold)
                    preds = []
                    probas = []

                    # SAVE TRAIN AND TEST
                    filename = "%i_%i_%i_%s_%s" % (repeat, q_id, fold, key, i_s[i])

                    print("B", train.shape, test.shape)
                    train = np.load("cozio/tra_%s.npy" % filename)
                    test = np.load("cozio/tes_%s.npy" % filename)

                    for n_start in range(n_range):
                        for n_end in range(n_range):
                            if n_start <= n_end:
                                n_ran = (n_start+1, n_end+1)

                                filename = "%i_%i_%i_%s_%s_%i_%i" % (repeat, q_id, fold, key, i_s[i], *n_ran)

                    preds = np.load("predictions/%s.npy" % filename[:-4])
                    probas = np.load("probas/%s.npy" % filename[:-4])

                    print(preds.shape, probas.shape)
                    print(filename[:-4], "%.3f" % np.std(preds), preds.shape, probas.shape)

                    n_idxx = 0
                    for n_start in range(n_range):
                        for n_end in range(n_range):
                            if n_start <= n_end:
                                y_pred = preds[n_idxx]
                                proba = probas[n_idxx]

                                score = balanced_accuracy_score(y[resampled][test], y_pred)

                                #print("%.3f | %s" % (score, filename), y_pred.shape)


                                # KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
                                scores[key_id, i, repeat, fold, q_id,
                                       n_start, n_end] = score
                                probas_array[key_id, i, repeat, fold, q_id,
                                       n_start, n_end] = proba
                                ytest_array[repeat, fold, q_id] = y[resampled][test]

                                n_idxx += 1


                # exit()
                resampled = None

np.save("results/green", scores)
np.save("results/green_probas", probas_array)
np.save("results/green_ytest", ytest_array)
