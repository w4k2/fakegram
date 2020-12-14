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
    for extractor_id, extractor in enumerate(i_s):
        print("## Extractors: %s" % extractor)
        plot_scores = []
        for q_id, quantity in enumerate(quantities):
            print("# Quantity: %s" % quantity)
            y_test = y_true[:, :, q_id].reshape(10,)
            words_probas = probas[key_id, extractor_id, :, :, q_id].reshape(10,)

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
        plt.title("%s %s" % (key, extractor))
        plt.legend()
        plt.grid(ls="--", color=(0.85, 0.85, 0.85))
        plt.tight_layout()
        plt.savefig("figures/%s_%s_redplot.png" % (key, extractor), dpi=200)


# Plot of words and structs
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
plot_scores = []
for q_id, quantity in enumerate(quantities):
    print("## Quantity: %s" % quantity)
    y_test = y_true[:, :, q_id].reshape(10,)
    txt_words = probas[0, 0, :, :, q_id].reshape(10,)
    aut_words = probas[1, 0, :, :, q_id].reshape(10,)
    ttl_words = probas[2, 0, :, :, q_id].reshape(10,)
    e_words = np.mean(probas[:, 0, :, :, q_id], axis=0).reshape(10,)

    txt_struct = probas[0, 1, :, :, q_id].reshape(10,)
    aut_struct = probas[1, 1, :, :, q_id].reshape(10,)
    ttl_struct = probas[2, 1, :, :, q_id].reshape(10,)
    e_struct = np.mean(probas[:, 1, :, :, q_id], axis=0).reshape(10,)

    e = np.mean(probas[:, :, :, :, q_id], axis=(0,1)).reshape(10,)

    fold_scores=np.zeros((9,10))
    for i in range(10):
        pred_txt_words = np.argmax(txt_words[i], axis=1)
        pred_aut_words = np.argmax(aut_words[i], axis=1)
        pred_ttl_words = np.argmax(ttl_words[i], axis=1)
        pred_e_words = np.argmax(e_words[i], axis=1)

        pred_txt_struct = np.argmax(txt_struct[i], axis=1)
        pred_aut_struct = np.argmax(aut_struct[i], axis=1)
        pred_ttl_struct = np.argmax(ttl_struct[i], axis=1)
        pred_e_struct = np.argmax(e_struct[i], axis=1)

        pred_e = np.argmax(e[i], axis=1)

        fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_txt_words)
        fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_aut_words)
        fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_ttl_words)
        fold_scores[3,i] = balanced_accuracy_score(y_test[i], pred_e_words)

        fold_scores[4,i] = balanced_accuracy_score(y_test[i], pred_txt_struct)
        fold_scores[5,i] = balanced_accuracy_score(y_test[i], pred_aut_struct)
        fold_scores[6,i] = balanced_accuracy_score(y_test[i], pred_ttl_struct)
        fold_scores[7,i] = balanced_accuracy_score(y_test[i], pred_e_struct)

        fold_scores[8,i] = balanced_accuracy_score(y_test[i], pred_e)
    plot_scores.append(np.mean(fold_scores, axis=1))

plot_scores = np.array(plot_scores).T
labels = ['text_w', 'author_w', 'title_w', 'ensemble_w', 'text_s', 'author_s', 'title_s', 'ensemble_s', 'ensemble_all']
ls = [":", "--", "-.", "-", ":", "--", "-.", "-", "-"]
colors = ["blue", "blue", "blue", "blue", "red", "red", "red", "red", "black"]
fig = plt.figure(figsize=(10, 5))
for i in range(plot_scores.shape[0]):
    plt.plot(plot_scores[i], color=colors[i], ls=ls[i], label=labels[i])

    # Annotate
    xytext = [(0,5), (0,5), (0,5), (0,5), (0,5), (0,5), (0,-10), (0,5), (0,10)]
    for j in range(plot_scores[i].shape[0]):
        plt.annotate(str(round(plot_scores[i][j], 3)), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=6)
#        plt.annotate((str(round(plot_scores[i][j], 3))+"+"+str(round(stds[i][j], 3))), (j,plot_scores[i][j]), textcoords="offset points", xytext=xytext[i], ha='center', fontsize=4)


plt.ylim(0.5, 1.0)
plt.title("Ensamble of 6")
plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
plt.title("Ensemble of 6")
plt.legend()
plt.grid(ls="--", color=(0.85, 0.85, 0.85))
plt.tight_layout()
plt.savefig("figures/ensemble6_redplot.png", dpi=200)
plt.savefig("figures/ensemble6_redplot.eps")
