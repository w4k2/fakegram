"""
To jest ten plik, co robi zielone tabelki.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score

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

print(y, y.shape)

s_idx = np.array(range(len(y))).astype(int)
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)

# Extraction loop
for key in keys:
    print("# Key %s" % key)
    for i, base in enumerate(i_s):
        print("## Base %s" % base)
        for repeat in range(n_repeats):
            for q_id, quantity in enumerate(quantities):
                print("#### Quantity %.2f [%i]" % (quantity, q_id))
                # Quantity resampling

                greentable = np.zeros((n_range, n_range))

                print("### Repeat %i" % repeat)
                resampled = resample(s_idx, n_samples=int(len(y)*quantity),
                                     replace=False, stratify=y,
                                     random_state=random_state + repeat)

                for fold, (train, test) in enumerate(skf.split(y[resampled],
                                                               y[resampled])):
                    print("##### Fold %i" % fold)


                    filename = "%i_%i_%i_%s_%s" % (repeat, q_id, fold, key, i_s[i])


                    preds = np.load("predictions/%s.npy" % filename)
                    probas = np.load("probas/%s.npy" % filename)


                    ngram_idx = 0
                    # print("HERE, KURWA!", preds.shape)
                    for n_start in range(n_range):
                        for n_end in range(n_range):
                            if n_start <= n_end:
                                # print(n_start, n_end, "TU IDX %i" % ngram_idx)
                                y_pred = preds[ngram_idx]
                                score = balanced_accuracy_score(y[resampled][test],y_pred)

                                greentable[n_start, n_end] += score

                                ngram_idx += 1

            print(greentable)
            exit()
