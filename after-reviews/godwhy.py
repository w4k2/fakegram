import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

df_af = pd.read_csv('../data/data_struct.csv')
df_raw = pd.read_csv('../data/data.csv')
X_af = df_af['title'].to_numpy().astype('U')
X_raw = df_raw['title'].to_numpy().astype('U')
y = df_af['label'].values.astype(int)

print(X_af, X_af.shape)
print(X_raw, X_raw.shape)
print(y, y.shape)

"""
Loop
"""
n_splits = 2
quantity = .4
s_idx = np.array(range(len(y))).astype(int)

skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)

resampled = resample(s_idx, n_samples=int(len(y)*quantity),
                     replace=False, stratify=y,
                     random_state=1410)

print(resampled.shape)

for fold, (train, test) in enumerate(skf.split(y[resampled],
                                               y[resampled])):
    print(fold, train.shape, test.shape)

    extractor_af = CountVectorizer(max_features=100, ngram_range=(2,2))
    extractor_raw = CountVectorizer(max_features=100, ngram_range=(2,2))

    extractor_af.fit(X_af[resampled][train])
    extractor_raw.fit(X_raw[resampled][train])

    X_af_t = extractor_af.transform(X_af[resampled])
    X_raw_t = extractor_raw.transform(X_raw[resampled])

    print("TRAIN AF")
    clf_af = MLPClassifier(random_state=1410).fit(X_af_t[train], y[resampled][train])

    print("TRAIN RAW")
    clf_raw = MLPClassifier(random_state=1410).fit(X_raw_t[train], y[resampled][train])

    y_preds = np.array([
        clf_af.predict(X_af_t[test]),
        clf_raw.predict(X_raw_t[test]),
        y[resampled][test]
    ])

    print(y_preds.shape)

    np.save('gggod', y_preds)

    #clf_af = MLPClassifier(random_state=1410).fit(X_af_t[train])

    exit()
