import numpy as np
import weles as ws
from tabulate import tabulate

all_scores = np.load("results/blue_stat.npy", allow_pickle=True)

what = ["text", "author", "title", "words", "struct", "all"]
# (0)txt_w, (1)aut_w, (2)ttl_w, (3)txt_s, (4)aut_s, (5)ttl_s, (6)e_w, (7)e_s, (8)e, (9)e_txt, (10)e_aut, (11)e_ttl
table_scores = np.concatenate((all_scores[5], [all_scores[0][2]], [all_scores[1][2]],[all_scores[2][2]]), axis=0)
# (0)txt_w, (1)aut_w, (2)ttl_w, (3)txt_s, (4)aut_s, (5)ttl_s, (6)e_w, (7)e_s, (8)e_txt, (9)e_aut, (10)e_ttl, (11)e
table_scores[[8, 11]] = table_scores[[11, 8]]
table_scores[[8, 9]] = table_scores[[9, 8]]
table_scores[[9, 10]] = table_scores[[10, 9]]

print(table_scores, table_scores.shape)

print("\n######## PAIRED TEST 40% ########\n")
headers = ["(1)txt_w", "(2)aut_w", "(3)ttl_w", "(4)txt_s", "(5)aut_s", "(6)ttl_s", "(7)e_w", "(8)e_s", "(9)e_txt", "(10)e_aut", "(11)e_ttl", "(12)e"]
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

table = tabulate(t, headers=headers)
print(table)
