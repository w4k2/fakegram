import numpy as np
import matplotlib.pyplot as plt

keys = ['text', 'author', 'title']
i_s = ['words', 'struct']
n_range = 6
ranger = np.array(range(n_range)) + 1
# quantities = np.array([.02, .05, .1, .25, .40])


# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.load("results/green.npy")

# KEYS x EXTRACTOR x QUANTITIES x FROM x TO
scores = np.mean(scores, axis=(2,3))

q_id = 4
quantity = 0.4
n_from = []
n_to = []
for key_id, key in enumerate(keys):
    for extractor_id, i in enumerate(i_s):
        print(q_id, key_id, extractor_id)

        green_score = scores[key_id, extractor_id, q_id]
        print(green_score)
        best_clf = np.argwhere(green_score==green_score.max())
        print("NAJLEPSZY:\n", best_clf)
        n_from.append(best_clf[0,0])
        n_to.append(best_clf[0,1])

print(n_from)
print(n_to)
# exit()
# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
probas = np.load("results/green_probas.npy", allow_pickle=True)
y_true = np.load("results/green_ytest.npy", allow_pickle=True)

best = 0
for key_id, key in enumerate(keys):
    for extractor_id, i in enumerate(i_s):
        for repeat_id in range(5):
            for fold_id in range(2):
                print(repeat_id, fold_id, q_id, key_id, extractor_id)
                print(n_from[best], n_to[best])

                green_proba = probas[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
                y_test = y_true[key_id, extractor_id, repeat_id, fold_id, q_id, n_from[best], n_to[best]]
                # print(y_test)
                # print(np.argmax(green_proba, axis=1))
                # best_clf = np.argwhere(green_score==green_score.max())
                # print("NAJLEPSZY:\n", best_clf)
        best+=1
