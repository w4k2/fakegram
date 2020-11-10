import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


keys = ['text', 'author', 'title']
i_s = ['words', 'struct']
n_range = 6
ranger = np.array(range(n_range)) + 1
# quantities = np.array([.02, .05, .1, .25, .40])


# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.load("results/green.npy")

# KEYS x EXTRACTOR x QUANTITIES x FROM x TO
scores = np.mean(scores, axis=(2,3))

# Finding best ngram range
quantities = np.array([.02, .05, .1, .25, .40])
n_from = []
n_to = []
for q_id, quantity in enumerate(quantities):
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):
            # print(q_id, key_id, extractor_id)

            green_score = scores[key_id, extractor_id, q_id]
            # print(green_score)
            best_clf = np.argwhere(green_score==green_score.max())
            # print("NAJLEPSZY:\n", best_clf)
            n_from.append(best_clf[0,0])
            n_to.append(best_clf[0,1])

# print(n_from)
# print(n_to)
# exit()
# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
probas = np.load("results/green_probas.npy", allow_pickle=True)
y_true = np.load("results/green_ytest.npy", allow_pickle=True)


# authors words + struct
zbiornik_proba = np.zeros((2, 10), dtype=object)
zbiornik_true = np.zeros((2, 10), dtype=object)

# best = 0
# for key_id, key in enumerate(keys):
#     for extractor_id, i in enumerate(i_s):
#         fold = 0
#         for repeat_id in range(5):
#             for fold_id in range(2):
#                 if key == 'text':
#                 # print(repeat_id, fold_id, q_id, key_id, extractor_id)
#                     # print(n_from[best], n_to[best])
#
#                     green_proba = probas[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
#                     y_test = y_true[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
#                     zbiornik_proba[extractor_id, fold] = green_proba
#                     zbiornik_true[extractor_id, fold] = y_test
#                     fold+=1
#
#                 # print(y_test)
#                 # print(np.argmax(green_proba, axis=1))
#                 # best_clf = np.argwhere(green_score==green_score.max())
#                 # print("NAJLEPSZY:\n", best_clf)
#         best+=1
#
#
# probas_all = np.mean(zbiornik_proba, axis=0)
# probas_words = zbiornik_proba[0]
# probas_struct = zbiornik_proba[1]
# # print(probas_words)
# scores_all = []
# scores_w = []
# scores_s = []
# # [print(np.argmax(probas[i], axis=1)) for i in range(probas.shape[0])]
#
# for i in range(probas_all.shape[0]):
#     pred = np.argmax(probas_all[i], axis=1)
#     pred_words = np.argmax(probas_words[i], axis=1)
#     pred_struct = np.argmax(probas_struct[i], axis=1)
#     true = zbiornik_true[0, i]
#     score = balanced_accuracy_score(true, pred)
#     scores_all.append(score)
#     score_w = balanced_accuracy_score(true, pred_words)
#     scores_w.append(score_w)
#     score_s = balanced_accuracy_score(true, pred_struct)
#     scores_s.append(score_s)
#
# print("text:")
# print("Words: %.3f | %.3f" % (np.mean(scores_w), np.std(scores_w)))
# print("Struct: %.3f | %.3f" % (np.mean(scores_s), np.std(scores_s)))
# print("Ensemble: %.3f | %.3f" % (np.mean(scores_all), np.std(scores_all)))


"""
Walka
"""

best = 0
plot_scores = np.zeros((3, 3, 5))
for q_id, quantity in enumerate(quantities):
    # authors words + struct
    zbiornik_proba = np.zeros((2, 10), dtype=object)
    zbiornik_true = np.zeros((2, 10), dtype=object)
    print("Quantity: %.2f" % quantity)
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):
            fold = 0
            for repeat_id in range(5):
                for fold_id in range(2):
                        # print(best)
                        green_proba = probas[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
                        y_test = y_true[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
                        zbiornik_proba[extractor_id, fold] = green_proba
                        zbiornik_true[extractor_id, fold] = y_test
                        fold+=1
            best+=1
        probas_all = np.mean(zbiornik_proba, axis=0)
        probas_words = zbiornik_proba[0]
        probas_struct = zbiornik_proba[1]
        # print(probas_words)
        scores_all = []
        scores_w = []
        scores_s = []
        # [print(np.argmax(probas[i], axis=1)) for i in range(probas.shape[0])]

        for i in range(probas_all.shape[0]):
            pred = np.argmax(probas_all[i], axis=1)
            pred_words = np.argmax(probas_words[i], axis=1)
            pred_struct = np.argmax(probas_struct[i], axis=1)
            true = zbiornik_true[0, i]
            score = balanced_accuracy_score(true, pred)
            scores_all.append(score)
            score_w = balanced_accuracy_score(true, pred_words)
            scores_w.append(score_w)
            score_s = balanced_accuracy_score(true, pred_struct)
            scores_s.append(score_s)

        print("%s:" % key)
        print("Words: %.3f | %.3f" % (np.mean(scores_w), np.std(scores_w)))
        print("Struct: %.3f | %.3f" % (np.mean(scores_s), np.std(scores_s)))
        print("Ensemble: %.3f | %.3f" % (np.mean(scores_all), np.std(scores_all)))

        plot_scores[key_id, 0, q_id] = np.mean(scores_w)
        plot_scores[key_id, 1, q_id] = np.mean(scores_s)
        plot_scores[key_id, 2, q_id] = np.mean(scores_all)
print(plot_scores)


labels = ['words', 'struct', 'ensemble']
ls = [":", "--", "-"]
for key_id, key in enumerate(keys):
    # print(key)
    scores = plot_scores[key_id]
    # print(scores)
    fig = plt.figure(figsize=(10, 5))
    for i in range(scores.shape[0]):
        plt.plot(scores[i], color='blue', ls=ls[i], label=labels[i])

    plt.ylim(0.5, 1.0)
    plt.xticks([i for i in range(quantities.shape[0])], [str(type) for type in quantities])
    plt.xlim(0, 4)
    plt.title(key)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/%s_lineplot.png" % key, dpi=200)
    # exit()
