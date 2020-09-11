import numpy as np
import matplotlib.pyplot as plt

data = np.load("n_gram_scores.npy")

print(data)

exit()

fig, ax = plt.subplots(1,1, figsize=(8,5))

a = np.array([0,1,2])
ax.bar(a-.125, data[0], width=.125, color="black", label = "FEATURES")
ax.bar(a+.125, data[1], width=.125, color="red", label="SUM(FEATURES)")

for i, v in enumerate(data[0]):
    ax.text(i-.125, v+.01, "%.3f" % v if v < 1000 else ("%.1fk" % (
        v / 1000)), color='black', ha='center', va='bottom')

for i, v in enumerate(data[1]):
    ax.text(i+.125, v+.01, "%.3f" % v if v < 1000 else ("%.1fk" % (
        v / 1000)), color='red', ha='center', va='bottom')


ax.set_xticks(a)
ax.set_xticklabels(["STRUCT", "WORDS", "ENSEMBLE"])
ax.set_ylim(.5,1)
ax.set_xlim(-.5,2.5)
plt.legend(ncol=3, loc=9, frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.grid(ls=":")
plt.tight_layout()

plt.savefig("sumplot.eps")
plt.savefig("sumplot.png")
