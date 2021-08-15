import numpy as np
from strlearn.metrics import balanced_accuracy_score


keys = ['text', 'author', 'title']
n_splits = 2
n_repeats = 5

y = np.load("all_y_test_old.npy")
y = y.reshape(n_repeats, n_splits, y.shape[1])

# Keys
for key in keys:
    print("%s fold scores:" % key)
    fold_scores = []
    for repeat in range(n_repeats):
        y_repeat = y[repeat]
        proba = np.load("probas/%i_%s_old.npy" % (repeat, key))
        pred = np.argmax(proba, axis=2)

        for split in range(n_splits):
            fold_score = balanced_accuracy_score(y_repeat[split], pred[split])
            fold_scores.append(fold_score)
            print("%.3f" % fold_score)
    fold_scores = np.array(fold_scores)
    key_mean_score = np.mean(fold_scores)
    print("%s mean score: %.3f" % (key, key_mean_score))
    print("\n")

# Ensemble
fold_scores = []
print("Ensemble fold scores:")
for repeat in range(n_repeats):
    y_repeat = y[repeat]

    probas = []
    for key in keys:
        proba = np.load("probas/%i_%s_old.npy" % (repeat, key))
        probas.append(proba)
    probas = np.array(probas)
    proba_mean = np.mean(probas, axis=0)
    pred = np.argmax(proba_mean, axis=2)

    for split in range(n_splits):
        fold_score = balanced_accuracy_score(y_repeat[split], pred[split])
        fold_scores.append(fold_score)
        print("%.3f" % fold_score)

fold_scores = np.array(fold_scores)
ensemble_mean_score = np.mean(fold_scores)
print("Ensemble mean score: %.3f" % (ensemble_mean_score))
print("\n")
