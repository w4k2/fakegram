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
    print("### Key: %s" % key)
    for extractor_id, extractor in enumerate(i_s):
        print("## Extractors: %s" % extractor)
        plot_scores = []
        std_scores = []
        for q_id, quantity in enumerate(quantities):
            print("# Quantity: %s" % quantity)
            y_test = y_true[:, :, q_id].reshape(10,)
            probas_16 = probas[key_id, extractor_id, :, :, q_id, 1].reshape(10,)

            fold_scores = np.zeros(10)
            for i in range(10):
                pred_probas_16 = np.argmax(probas_16[i], axis=1)
                fold_scores[i] = balanced_accuracy_score(y_test[i], pred_probas_16)
            plot_scores.append(np.mean(fold_scores))
            std_scores.append(np.std(fold_scores))
        plot_scores = np.array(plot_scores).T

        fig = plt.figure(figsize=(10, 5))
        plt.plot(plot_scores, color='orange', label=extractor)

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
        plt.savefig("figures/%s_%s_16_orangeplot.png" % (key, extractor), dpi=200)

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
# Results by key for diagonal - struct and words
print("### Diagonal ###")
for key_id, key in enumerate(keys):
    words_diagonal = np.zeros(n_range, dtype=object)
    struct_diagonal = np.zeros(n_range, dtype=object)
    print("## Key: %s" % key)
    for i in range(n_range):
        words_diagonal[i] = probas[key_id, 0, :, :, 4, i, i]
        struct_diagonal[i] = probas[key_id, 1, :, :, 4, i, i]
    words_diagonal = np.mean(words_diagonal).reshape(10,)
    struct_diagonal = np.mean(struct_diagonal).reshape(10,)

    fold_scores = np.zeros((2,10))
    for i in range(10):
        pred_words_diagonal = np.argmax(words_diagonal[i], axis=1)
        pred_struct_diagonal = np.argmax(struct_diagonal[i], axis=1)
        fold_scores[0, i] = balanced_accuracy_score(y_test[i], pred_words_diagonal)
        fold_scores[1, i] = balanced_accuracy_score(y_test[i], pred_struct_diagonal)
    print("Scores: %s" % np.mean(fold_scores))
    print("Standard deviation: %s" % np.std(fold_scores))

# TO CZEGO SZUKASZ JEST TUTAJ
# Ensemble diagonal
e_diagonal_words = np.mean(probas[:, 0, :, :, 4, :, :], axis=0)
e_diagonal_words_array = np.zeros(n_range, dtype=object)

e_diagonal_struct = np.mean(probas[:, 1, :, :, 4, :, :], axis=0)
e_diagonal_struct_array = np.zeros(n_range, dtype=object)

e_diagonal_ensemble = np.mean(probas[:, :, :, :, 4, :, :], axis=(0,1))
e_diagonal_ensemble_array = np.zeros(n_range, dtype=object)

for i in range(n_range):
    e_diagonal_words_array[i] = e_diagonal_words[:, :, i, i]
    e_diagonal_struct_array[i] = e_diagonal_struct[:, :, i, i]
    e_diagonal_ensemble_array[i] = e_diagonal_ensemble[:, :, i, i]

e_diagonal_words = np.mean(e_diagonal_words_array).reshape(10,)
e_diagonal_struct = np.mean(e_diagonal_struct_array).reshape(10,)
e_diagonal_ensemble = np.mean(e_diagonal_ensemble_array).reshape(10,)

fold_scores = np.zeros((3,10))
for i in range(10):
    pred_e_diagonal_words = np.argmax(e_diagonal_words[i], axis=1)
    pred_e_diagonal_struct = np.argmax(e_diagonal_struct[i], axis=1)
    pred_e_diagonal_ensemble = np.argmax(e_diagonal_ensemble[i], axis=1)

    fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_e_diagonal_words)
    fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_e_diagonal_struct)
    fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_e_diagonal_ensemble)

print("## Ensemble")
print("Words scores: %s" % np.mean(fold_scores[0]))
print("Struct scores: %s" % np.mean(fold_scores[1]))
print("Ensemble cores: %s" % np.mean(fold_scores[2]))
