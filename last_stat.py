import numpy as np
import weles as ws
from tabulate import tabulate

table_scores = np.load("final_plots_scores.npy")

what = ["UNI_WORDS", "UNI_STRUCT", "BI_WORDS", "BI_STRUCT",  "B_WORDS", "B_STRUCT", "B_ALL", "O_WORDS", "O_STRUCT", "O_ALL"]

# print(table_scores, table_scores.shape)

print("\n######## PAIRED TEST 40% ########\n")
# headers = ["(1)txt_w", "(2)aut_w", "(3)ttl_w", "(4)txt_s", "(5)aut_s", "(6)ttl_s", "(7)e_w", "(8)e_s", "(9)e_txt", "(10)e_aut", "(11)e_ttl", "(12)e"]
t = []
n_clfs = table_scores.shape[0]
alpha = .05
# fold, clf, metric
data_scores = table_scores.T

mean_scores = np.squeeze(np.mean(data_scores, axis=0).T)
stds = np.squeeze(np.std(data_scores, axis=0).T)
t.append(["BAC"] + ["%.3f" % v for v in mean_scores])
t.append(["std"] + ["%.3f" % v for v in stds])

T, p = np.array([[ws.statistics.t_test_corrected(data_scores[:, i], data_scores[:, j], J=5, k=2)
                  for i in range(n_clfs)] for j in range(n_clfs)]).swapaxes(0, 2)
_ = np.where((p < alpha) * (T > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
t.append([''] + [", ".join(["%i" % i for i in c]) if len(c) > 0 and len(c) < n_clfs -
                 1 else ("all" if len(c) == n_clfs - 1 else "---") for c in conclusions])

table = tabulate(t, headers=what)
print(table)
