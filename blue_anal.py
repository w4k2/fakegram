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
Finding best n-gram range
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
        for i in range(10):
            pred_w = np.argmax(w[i], axis=1)
            pred_s = np.argmax(s[i], axis=1)
            pred_e = np.argmax(e[i], axis=1)
            fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_w)
            fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_s)
            fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_e)
        plot_scores.append(np.mean(fold_scores, axis=1))
    plot_scores = np.array(plot_scores).T

    # Lets make some plots
    labels = ['words', 'struct', 'ensemble']
    ls = [":", "--", "-"]
    fig = plt.figure(figsize=(10, 5))
    for i in range(plot_scores.shape[0]):
        plt.plot(plot_scores[i], color='blue', ls=ls[i], label=labels[i])
        # Annotate
        xytext = [(0,-10), (0,10), (0,10)]
        for j in range(plot_scores[i].shape[0]):
            plt.annotate(str(round(plot_scores[i][j], 3)), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=6)

    plt.ylim(0.5, 1.0)
    plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
    # plt.xlim(0, 4)
    plt.title(key)
    plt.legend()
    plt.grid(ls="--", color=(0.85, 0.85, 0.85))
    plt.tight_layout()
    plt.savefig("figures/%s_blueplot.png" % key, dpi=200)


"""
Plots by extractor
"""
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
for extractor_id, extractor in enumerate(i_s):
    print(extractor)
    plot_scores = []
    for q_id, quantity in enumerate(quantities):
        y_test = y_true[:, :, q_id].reshape(10,)
        separate_proba = best_probas[q_id, :, extractor_id]

        txt = separate_proba[0].reshape(10,)
        aut = separate_proba[1].reshape(10,)
        ttl = separate_proba[2].reshape(10,)
        e = np.mean(best_probas[q_id, :, extractor_id], axis=0).reshape(10,)

        # calculate mean scores from folds
        fold_scores = np.zeros((4 ,10))
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

    # Lets make some plots
    labels = ['text', 'author', 'title', 'ensemble']
    ls = [":", "--", "-.", "-"]
    fig = plt.figure(figsize=(10, 5))
    for i in range(plot_scores.shape[0]):
        plt.plot(plot_scores[i], color='blue', ls=ls[i], label=labels[i])
        # Annotate
        xytext = [(0,5), (0,5), (0,5), (0,5)]
        for j in range(plot_scores[i].shape[0]):
            plt.annotate(str(round(plot_scores[i][j], 3)), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=6)

    plt.ylim(0.5, 1.0)
    plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
    # plt.xlim(0, 4)
    plt.title(extractor)
    plt.legend()
    plt.grid(ls="--", color=(0.85, 0.85, 0.85))
    plt.tight_layout()
    plt.savefig("figures/%s_blueplot.png" % extractor, dpi=200)

"""
Ensemble of 6
"""
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
plot_scores = []
stds = []
for q_id, quantity in enumerate(quantities):
    y_test = y_true[:, :, q_id].reshape(10,)
    separate_proba = best_probas[q_id, :, :]

    txt_w = separate_proba[0,0].reshape(10,)
    aut_w = separate_proba[1,0].reshape(10,)
    ttl_w = separate_proba[2,0].reshape(10,)
    txt_s = separate_proba[0,1].reshape(10,)
    aut_s = separate_proba[1,1].reshape(10,)
    ttl_s = separate_proba[2,1].reshape(10,)
    e = np.mean(best_probas[q_id, :, :], axis=(0,1)).reshape(10,)
    e_w = np.mean(best_probas[q_id, :, 0], axis=0).reshape(10,)
    e_s = np.mean(best_probas[q_id, :, 1], axis=0).reshape(10,)

    # calculate mean scores from folds
    fold_scores = np.zeros((9 ,10))
    for i in range(10):
        pred_txt_w = np.argmax(txt_w[i], axis=1)
        pred_aut_w = np.argmax(aut_w[i], axis=1)
        pred_ttl_w = np.argmax(ttl_w[i], axis=1)
        pred_txt_s = np.argmax(txt_s[i], axis=1)
        pred_aut_s = np.argmax(aut_s[i], axis=1)
        pred_ttl_s = np.argmax(ttl_s[i], axis=1)
        pred_e = np.argmax(e[i], axis=1)
        pred_e_w = np.argmax(e_w[i], axis=1)
        pred_e_s = np.argmax(e_s[i], axis=1)
        fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_txt_w)
        fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_aut_w)
        fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_ttl_w)
        fold_scores[3,i] = balanced_accuracy_score(y_test[i], pred_txt_s)
        fold_scores[4,i] = balanced_accuracy_score(y_test[i], pred_aut_s)
        fold_scores[5,i] = balanced_accuracy_score(y_test[i], pred_ttl_s)
        fold_scores[8,i] = balanced_accuracy_score(y_test[i], pred_e)
        fold_scores[6,i] = balanced_accuracy_score(y_test[i], pred_e_w)
        fold_scores[7,i] = balanced_accuracy_score(y_test[i], pred_e_s)
    plot_scores.append(np.mean(fold_scores, axis=1))
    stds.append(np.std(fold_scores, axis=1))
plot_scores = np.array(plot_scores).T
stds = np.array(stds).T

# Lets make some plots
labels = ['text_w', 'author_w', 'title_w','text_s', 'author_s', 'title_s', 'ensemble_w', 'ensemble_s', 'ensemble_all']
ls = [":", "--", "-.",":", "--", "-.", "-", "-", "-"]
colors = ["blue", "blue", "blue", "red", "red", "red", "blue", "red", "black"]
alphas = [.4,.4,.4,.4,.4,.4,1.0,1.0,1.0]
fig = plt.figure(figsize=(10, 5))
for i in range(plot_scores.shape[0]):
    plt.plot(plot_scores[i], color=colors[i], ls=ls[i], label=labels[i], alpha=alphas[i])
    # # Annotate
    xytext = [(0,5), (0,5), (0,5), (0,5), (0,5), (0,5), (0,-10), (0,5), (0,10)]
    for j in range(plot_scores[i].shape[0]):
        if i == 6 or i == 7 or i == 8:
            plt.annotate((str(round(plot_scores[i][j], 3))+"+"+str(round(stds[i][j], 3))), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=4)

plt.ylim(0.5, 1.0)
plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
# plt.xlim(0, 4)
plt.title("Ensemble of 6")
plt.legend()
plt.grid(ls="--", color=(0.85, 0.85, 0.85))
plt.tight_layout()
plt.savefig("figures/ensemble6_blueplot.png", dpi=200)
