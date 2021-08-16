import numpy as np
import pandas as pd
from strlearn.metrics import balanced_accuracy_score


keys = ['text', 'title']
i_s = ['words', 'struct']
n_range = 6
n_splits = 2
n_repeats = 5
# Only 40%
# quantities = np.array([.02, .05, .1, .25, .40])


# KEYS x BASE x REPEATS x SPLITS x MODELS x SAMPLES x CLASSES
gathered = np.zeros((len(keys), len(i_s), n_repeats, n_splits, 21, 4000, 2))

for key_id, key in enumerate(keys):
    print("# Key %s" % key)
    for i, base in enumerate(i_s):
        print("## Base %s" % base)
        for repeat in range(n_repeats):
            print("### Repeat %i" % repeat)
            for fold in range(n_splits):
                # MODELS x SAMPLES x CLASSES
                proba = np.load('probas_new/%i_%i_%i_%s_%s.npy' % (repeat, 4, fold, key, i_s[i]))
                gathered[key_id, i, repeat, fold] = proba

print(gathered.shape)

# ALL ensemble
# REPEATS x SPLITS x SAMPLES x CLASSES
mean_proba = np.mean(gathered, axis=(0,1,4))
print(mean_proba.shape)
# REPEATS x SPLITS x SAMPLES
pred = np.argmax(mean_proba, axis=3)
print(pred.shape)

# Scores
y = np.load("all_y_new_af.npy")
pred = pred.reshape(n_repeats * n_splits, pred.shape[2])
print(y.shape)
print(pred.shape)

fold_scores = []
print("Ensemble fold scores:")
for fold in range(pred.shape[0]):
    score = balanced_accuracy_score(y[fold], pred[fold])
    fold_scores.append(score)
    print("%.3f" % score)

fold_scores = np.array(fold_scores)
ensemble_mean_score = np.mean(fold_scores)
print("Ensemble mean score: %.3f" % (ensemble_mean_score))
