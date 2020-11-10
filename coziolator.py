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
                # SAVE RESAMPLED
                print("A", resampled.shape)
                filename = "%i_%i_%s_%s" % (repeat, q_id, key, i_s[i])
                np.save("cozio/res_%s" % filename, resampled)

                # print(resampled.shape)
                for fold, (train, test) in enumerate(skf.split(y[resampled],
                                                               y[resampled])):
                    print("##### Fold %i" % fold)
                    # SAVE TRAIN AND TEST
                    filename = "%i_%i_%i_%s_%s" % (repeat, q_id, fold, key, i_s[i])

                    print("B", train.shape, test.shape)
                    np.save("cozio/tra_%s" % filename, train)
                    np.save("cozio/tes_%s" % filename, test)

                resampled = None
