import numpy as np
import matplotlib.pyplot as plt

keys = ['text', 'author', 'title']
i_s = ['words', 'struct']
n_range = 6
ranger = np.array(range(n_range)) + 1
quantities = np.array([.02, .05, .1, .25, .40])


# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.load("results/green.npy")

print(scores.shape)

# KEYS x EXTRACTOR x QUANTITIES x FROM x TO
scores = np.mean(scores, axis=(2,3))

print(scores.shape)

for q_id, quantity in enumerate(quantities):
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):
            print(q_id, key_id, extractor_id)

            green_score = scores[key_id, extractor_id, q_id]

            print(green_score.shape)

            plt.clf()
            fig = plt.figure(figsize=(5,5))

            plt.imshow(green_score, cmap="Greens", vmin=.5, vmax=1)
            plt.xticks(range(n_range), ranger)
            plt.yticks(range(n_range), ranger)
            plt.ylabel("from")
            plt.xlabel("to")

            for start in range(n_range):
                for end in range(n_range):
                    if start >= end:
                        plt.text(start, end, "%.3f" % green_score[end,start],
                                 ha='center', va='center', c='black')

            filename = "k_%s_q_%s_e_%s" % (key, ("%.2f" % quantity)[2:], i)
            plt.title(filename, fontsize=10)
            plt.tight_layout()
            plt.savefig("foo.png")
            plt.savefig("figures/%s.png" % filename)
