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
# REPEATS x FOLDS x QUANTITIES -> REPEATS * FOLDS x QUANTITIES
y_true = np.load("results/green_ytest.npy", allow_pickle=True).reshape(10,5)

"""
Finding best ngram range
"""
# KEYS x EXTRACTOR x QUANTITIES x FROM x TO
scores = np.mean(scores, axis=(2,3))
# N_TABLES x x REPEATS x FOLDS

# QUANTITIES x KEYS x EXTRACTOR
best_probas = np.zeros((5, 3, 2), dtype=object)

n_from = []
n_to = []
for q_id, quantity in enumerate(quantities):
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):
            # print(q_id, key_id, extractor_id)
            green_score = scores[key_id, extractor_id, q_id]
            best_clf = np.argwhere(green_score==green_score.max())
            # print("NAJLEPSZY:\n", best_clf)
            n_from.append(best_clf[0,0])
            n_to.append(best_clf[0,1])
            best_probas[q_id, key_id, extractor_id] = probas[key_id, extractor_id, :, :, q_id, best_clf[0,0], best_clf[0,1]]

"""
Plots by key
"""
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
for key_id, key in enumerate(keys):
    print(key)
    plot_scores = []
    for q_id, quantity in enumerate(quantities):
        y_test = y_true[:, :, q_id].reshape(10,)
        separate_proba = best_probas[q_id, key_id, :]
        w = separate_proba[0].reshape(10,)
        s = separate_proba[1].reshape(10,)
        e = np.mean(best_probas[q_id, key_id, :], axis=0).reshape(10,)

        # calculate mean scores from folds
        fold_scores = np.zeros((3 ,10))
        for i in range(w.shape[0]):
            pred_w = np.argmax(w[i], axis=1)
            pred_s = np.argmax(s[i], axis=1)
            pred_e = np.argmax(e[i], axis=1)
            fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_w)
            fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_s)
            fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_e)
        # print(np.mean(fold_scores, axis=1))
        plot_scores.append(np.mean(fold_scores, axis=1))
        # exit()
    plot_scores = np.array(plot_scores).T

    # Lets make some plots
    labels = ['words', 'struct', 'ensemble']
    ls = [":", "--", "-"]
    fig = plt.figure(figsize=(10, 5))
    for i in range(plot_scores.shape[0]):
        plt.plot(plot_scores[i], color='blue', ls=ls[i], label=labels[i])

    plt.ylim(0.5, 1.0)
    plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
    plt.xlim(0, 4)
    plt.title(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/%s_blueplot.png" % key, dpi=200)
