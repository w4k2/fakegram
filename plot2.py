import numpy as np
import matplotlib.pyplot as plt

results_11 = np.load("e11_2.npy")
results_12 = np.load("e12_2.npy")
results_22 = np.load("e22_2.npy")
results_11 = np.mean(results_11, axis=1).T[:2]
results_12 = np.mean(results_12, axis=1).T[:2]
results_22 = np.mean(results_22, axis=1).T[:2]
quantities = np.array([.01, .02, .05, .1, .25, .50, .75, 1.])
quantities = np.array([.0004, .0036, .0068, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

print(results_11.shape)

labels = ["STRUCT", "WORDS"]
lss = ['--', ":", "-"]
colors = ['red', 'black']

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for i, label in enumerate(labels):
    print(i, label)
    ax.plot(quantities, results_11[i],
            label="%s (1,1)" % label, ls='--', c=colors[i])

for i, label in enumerate(labels):
    print(i, label)
    ax.plot(quantities, results_12[i],
            label="%s (1,2)" % label, ls=':', c=colors[i])


for i, label in enumerate(labels):
    print(i, label)
    ax.plot(quantities, results_22[i],
            label="%s (2,2)" % label, ls='-', c=colors[i])


ax.set_ylim(.5, 1)
ax.set_xlim(0,1)
ax.set_xlabel("Percentage of corpus used to establish model")
ax.set_ylabel("Balanced accuracy score")
ax.grid(ls=":")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(ncol=3, loc=8, frameon=False)

plt.tight_layout()
plt.savefig("plot2.png")
plt.savefig("plot2.eps")
