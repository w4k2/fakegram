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

probas = np.mean(probas, axis=(5,6))

# Plots by keys - only for words
for key_id, key in enumerate(keys):
    print("### Key: %s" % key)
    plot_scores = []
    for q_id, quantity in enumerate(quantities):
        print("## Quantity: %s" % quantity)
        y_test = y_true[:, :, q_id].reshape(10,)
        words_probas = probas[key_id, 0, :, :, q_id].reshape(10,)

        fold_scores = np.zeros(10)
        for i in range(10):
            pred_w = np.argmax(words_probas[i], axis=1)
            fold_scores[i] = balanced_accuracy_score(y_test[i], pred_w)
        plot_scores.append(np.mean(fold_scores))
    plot_scores = np.array(plot_scores).T

    fig = plt.figure(figsize=(10, 5))
    plt.plot(plot_scores, color='red', label='words')

    # Annotate
    xytext = [(0,-10), (0,10), (0,10)]
    for j in range(plot_scores.shape[0]):
        plt.annotate(str(round(plot_scores[j], 3)), (j,plot_scores[j]), textcoords="offset points", xytext=xytext[0], ha='center', fontsize=6)

    plt.ylim(0.5, 1.0)
    plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
    plt.title("%s words" % key)
    plt.legend()
    plt.grid(ls="--", color=(0.85, 0.85, 0.85))
    plt.tight_layout()
    plt.savefig("figures/%s_redplot.png" % key, dpi=200)


# Plot of words
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
plot_scores = []
for q_id, quantity in enumerate(quantities):
    print("## Quantity: %s" % quantity)
    y_test = y_true[:, :, q_id].reshape(10,)
    txt = probas[0, 0, :, :, q_id].reshape(10,)
    aut = probas[1, 0, :, :, q_id].reshape(10,)
    ttl = probas[2, 0, :, :, q_id].reshape(10,)
    e = np.mean(probas[:, 0, :, :, q_id], axis=0).reshape(10,)

    fold_scores=np.zeros((4,10))
    for i in range(10):
        pred_txt = np.argmax(txt[i], axis=1)
        pred_aut = np.argmax(aut[i], axis=1)
        pred_ttl = np.argmax(ttl[i], axis=1)
        pred_e = np.argmax(e[i], axis=1)
        fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_txt)
        fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_aut)
        fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_ttl)
        fold_scores[3,i] = balanced_accuracy_score(y_test[i], pred_e)
    plot_scores.append(np.mean(fold_scores, axis=1))

plot_scores = np.array(plot_scores).T
labels = ['text', 'author', 'title', 'ensamble']
ls = ["--", ":", "-.", "-"]
fig = plt.figure(figsize=(10, 5))
for i in range(plot_scores.shape[0]):
    plt.plot(plot_scores[i], color='red', ls=ls[i], label=labels[i])
    # Annotate
    xytext = [(0,5), (0,5), (0,5), (0,5)]
    for j in range(plot_scores[i].shape[0]):
        plt.annotate(str(round(plot_scores[i][j], 3)), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=6)

plt.ylim(0.5, 1.0)
plt.title("Ensamble")
plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
plt.legend()
plt.grid(ls="--", color=(0.85, 0.85, 0.85))
plt.tight_layout()
plt.savefig("figures/ensamble_redplot.png", dpi=200)
