import numpy as np
import matplotlib.pyplot as plt

results = np.load("e1.npy")
results = np.mean(results, axis=1).T
quantities = np.array([.01, .02, .05, .1, .25, .50, .75, 1.])
#quantities = np.array([.0004, .0036, .0068, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

print(results, results.shape)

labels = ["STRUCT", "WORDS", "ENSEMBLE"]
lss = ['-', "-", "-"]
css = ["blue", "red", "black"]

fig, ax = plt.subplots(1,1, figsize=(8,5))
for i, label in enumerate(labels):
    print(i, label)
    ax.plot(quantities, results[i], label=label, ls=lss[i], c=css[i])

ax.set_ylim(.5, 1)
ax.set_xlim(0,1)
ax.set_xlabel("Percentage of corpus used to establish model")
ax.set_ylabel("Balanced accuracy score")
ax.grid(ls=":")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(ncol=3, loc=8, frameon=False)

plt.tight_layout()
plt.savefig("foo.png")
plt.savefig("plot.png")
plt.savefig("plot.eps")
