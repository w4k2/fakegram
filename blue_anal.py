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
for key_id, key in enumerate(keys):
    for extractor_id, i in enumerate(i_s):
        print(q_id, key_id, extractor_id)

        green_score = scores[key_id, extractor_id, q_id]
        print(green_score)
        best_clf = np.argwhere(green_score==green_score.max())
        print("NAJLEPSZY:\n", best_clf)



# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
probas = np.load("results/green_probas.npy", allow_pickle=True)

for repeat_id in range(5):
    for fold_id in range(2):
        for key_id, key in enumerate(keys):
            for extractor_id, i in enumerate(i_s):
                print(repeat_id, fold_id, q_id, key_id, extractor_id)

                green_score = probas[key_id, extractor_id, repeat_id, fold_id, q_id, 0, 1]
                print(green_score)
                # best_clf = np.argwhere(green_score==green_score.max())
                # print("NAJLEPSZY:\n", best_clf)
