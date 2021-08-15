import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample

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
df_words = pd.read_csv('data/data_with_labels.csv')
df_words.columns = ['title', 'text', 'label']
df_struct = pd.read_csv('data/data_with_labels_struct.csv')
df_struct.columns = ['title', 'text', 'label']

y = df_words['label'].values.astype(int)

print("y", y, y.shape)

s_idx = np.array(range(len(y))).astype(int)

print(s_idx)
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)

# Extraction loop
for key in keys:
    # print("Key %s" % key)
    for i, base in enumerate([df_words[key], df_struct[key]]):
        X = base.values.astype('U')
        for repeat in range(n_repeats):
            print("# Repeat %i" % repeat)
            for q_id, quantity in enumerate(quantities):
                # print(q_id, quantity)
                # Quantity resampling
                resampled = resample(s_idx, n_samples=int(len(y)*quantity),
                                     replace=False, stratify=y,
                                     random_state=random_state + repeat)

                # print(resampled.shape)
                for fold, (train, test) in enumerate(skf.split(y[resampled],
                                                               y[resampled])):
                    # print("# Fold %i" % fold)
                    for n_start in range(n_range):
                        for n_end in range(n_range):
                            if n_start <= n_end:
                                n_ran = (n_start+1, n_end+1)
                                extractor = CountVectorizer(max_features=100,
                                                            ngram_range=n_ran)
                                extractor.fit(X[resampled][train])

                                X_transformed = extractor.transform(X[resampled])

                                filename = "%i_%i_%i_%s_%s_%i_%i" % (repeat, q_id, fold, key, i_s[i], *n_ran)

                                np.save("%s/%s" % ("extracted_new", filename),
                                        X_transformed)

                                print(filename, X_transformed.shape)
                                X_transformed = None
                resampled = None
        X = None
