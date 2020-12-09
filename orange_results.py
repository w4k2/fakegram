import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

keys = ['text', 'author', 'title']
i_s = ['words', 'struct']
n_range = 6
ranger = np.array(range(n_range)) + 1
quantities = np.array([.02, .05, .1, .25, .40])

# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.load("results/green.npy")
# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
probas = np.load("results/green_probas.npy", allow_pickle=True)
# REPEATS x FOLDS x QUANTITIES
y_true = np.load("results/green_ytest.npy", allow_pickle=True)

# KEYS x EXTRACTOR x QUANTITIES
scores = np.mean(scores, axis=(2,3))
probas = np.mean(probas, axis=6)

# Results by key for 1-6
print("### 1-6 ###")
for key_id, key in enumerate(keys):
    print("## Key: %s" % key)
    y_test = y_true[:, :, 4].reshape(10,)
    words_16 = probas[key_id, 0, :, :, 4, 1].reshape(10,)

    fold_scores = np.zeros(10)
    for i in range(10):
        pred_words_16 = np.argmax(words_16[i], axis=1)
        fold_scores[i] = balanced_accuracy_score(y_test[i], pred_words_16)
    print("Scores: %s" % np.mean(fold_scores))
    print("Standard deviation: %s" % np.std(fold_scores))

# Ensemble 1-6
e_16 = np.mean(probas[:, 0, :, :, 4, 1], axis=0).reshape(10,)
fold_scores=np.zeros(10)
for i in range(10):
    pred_e_16 = np.argmax(e_16[i], axis=1)
    fold_scores[i] = balanced_accuracy_score(y_test[i], pred_e_16)
print("## Ensemble ")
print("Scores: %s" % np.mean(fold_scores))
print("Standard deviation: %s" % np.std(fold_scores))
print()

probas = np.load("results/green_probas.npy", allow_pickle=True)
# Results by key for diagonal
print("### Diagonal ###")
for key_id, key in enumerate(keys):
    words_diagonal = np.zeros(n_range, dtype=object)
    print("## Key: %s" % key)
    for i in range(n_range):
        words_diagonal[i] = probas[key_id, 0, :, :, 4, i, i]
    words_diagonal = np.mean(words_diagonal).reshape(10,)

    fold_scores = np.zeros(10)
    for i in range(10):
        pred_words_diagonal = np.argmax(words_diagonal[i], axis=1)
        fold_scores[i] = balanced_accuracy_score(y_test[i], pred_words_diagonal)
    print("Scores: %s" % np.mean(fold_scores))
    print("Standard deviation: %s" % np.std(fold_scores))

# Ensemble diagonal
e_diagonal = np.mean(probas[:, 0, :, :, 4, :, :], axis=0)
e_diagonal_array = np.zeros(n_range, dtype=object)
for i in range(n_range):
    e_diagonal_array[i] = e_diagonal[:, :, i, i]

e_diagonal = np.mean(e_diagonal_array).reshape(10,)
fold_scores = np.zeros(10)
for i in range(10):
    pred_e_diagonal = np.argmax(e_diagonal[i], axis=1)
    fold_scores[i] = balanced_accuracy_score(y_test[i], pred_e_diagonal)
print("## Ensemble")
print("Scores: %s" % np.mean(fold_scores))
print("Standard deviation: %s" % np.std(fold_scores))
