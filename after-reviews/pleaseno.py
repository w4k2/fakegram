import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score as metric

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

    """
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
    """

    y_preds = np.load('gggod.npy')

    print(y_preds.shape)

    y_pred_af = y_preds[0]
    y_pred_raw = y_preds[1]
    y = y_preds[2]

    # Scores
    score_af = metric(y, y_pred_af)
    score_raw = metric(y, y_pred_raw)

    # Masks
    mask_agree = y_pred_af == y_pred_raw
    mask_disagree = y_pred_af != y_pred_raw

    mask_af_correct = y_pred_af == y
    mask_af_wrong = y_pred_af != y

    mask_raw_correct = y_pred_raw == y
    mask_raw_wrong = y_pred_raw != y

    mask_both_correct = mask_af_correct * mask_raw_correct
    mask_both_wrong = mask_af_wrong * mask_raw_wrong

    mask_af_correct_raw_wrong = mask_af_correct * mask_raw_wrong
    mask_af_wrong_raw_correct = mask_af_wrong * mask_raw_correct

    masks = {
        'same decisions': mask_agree,
        'different decisions': mask_disagree,
        'AF correct': mask_af_correct,
        'RAW correct': mask_raw_correct,
        'AF wrong': mask_af_wrong,
        'RAW wrong': mask_raw_wrong,
        'both correct': mask_both_correct,
        'both wrong': mask_both_wrong,
        'AF correct : RAW wrong': mask_af_correct_raw_wrong,
        'AF wrong : RAW correct': mask_af_wrong_raw_correct,
    }

    # Counts
    n = y.shape[0]
    n_examples = 20

    # Info
    print("SCORE _AF %.3f" % score_af)
    print("SCORE RAW %.3f" % score_raw)

    print("%5i samples" % n)

    for mname in masks:
        mask = masks[mname]
        count = np.sum(mask)
        print("\n# %5i %s" % (count, mname))

        indexes = np.where(mask)[0][:n_examples]

        for i, index in enumerate(indexes):
            print("%i%i%i AF: %s\n   RAW: %s" % (
                y[index], y_pred_af[index], y_pred_raw[index],
                X_af[test][index],
                X_raw[test][index],
            ))

    exit()
