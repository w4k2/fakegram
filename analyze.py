import numpy as np
import matplotlib.pyplot as plt
import weles as ws
from tabulate import tabulate

# kwanty, foldy, metody
scores = np.load("e1.npy")
# kwanty, metody
table_scores = np.mean(scores, axis=1)

# print(table_scores, table_scores.shape)

clfs = ["STRUCT", "WORDS", "ENSEMBLE"]
n_clfs = 3
metrics = ["bac"]
quants = ["01", "02", "05", "1", "25", "50", "75", "1"]
alpha = .05

# TESTY PAROWE
# print("\n######## PAIRED FOR EACH QUANT ########\n")

t = []
for q_id, q in enumerate(quants):
    # fold, clf, metric
    data_scores = scores[q_id, :, :]

    mean_scores = np.squeeze(np.mean(data_scores, axis=0).T)
    stds = np.squeeze(np.std(data_scores, axis=0).T)
    t.append(["%s" % q] + ["%.3f" % v for v in mean_scores])
    t.append(["std"] + ["%.3f" % v for v in stds])

    T, p = np.array([[ws.statistics.t_test_corrected(data_scores[:, i], data_scores[:, j], J=2, k=5)
                      for i in range(n_clfs)] for j in range(n_clfs)]).swapaxes(0, 2)
    _ = np.where((p < alpha) * (T > 0))
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
    t.append([''] + [", ".join(["%i" % i for i in c]) if len(c) > 0 and len(c) < len(clfs) -
                     1 else ("all" if len(c) == len(clfs) - 1 else "---") for c in conclusions])


table = tabulate(t, headers=clfs, tablefmt="latex_booktabs")
print(table)
