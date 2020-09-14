import numpy as np
import matplotlib.pyplot as plt

data = np.mean(np.load("n_gram_scores.npy"), axis=0)

a = data[0]
b = data[1]
# FOLDS x METHOD x I x J
print(data[0], data.shape)

fig, ax = plt.subplots(1,2, figsize=(8,4.5))

ax[0].imshow(a, cmap="Blues", vmax=1,vmin=.5, origin='lower')
ax[1].imshow(b, cmap="Reds", vmax=1,vmin=.5, origin='lower')
ax[0].set_title("STRUCT")
ax[1].set_title("WORDS")

for z in range(2):
    ax[z].set_xticks(range(5))
    ax[z].set_xticklabels(np.array(list(range(5)))+1)
    ax[z].set_yticks(range(5))
    ax[z].set_yticklabels(np.array(list(range(5)))+1)
    ax[z].set_ylabel("from")
    ax[z].set_xlabel("to")
for i, v in enumerate(a):
    for j, vv in enumerate(v):
        print(i, j, vv)
        if vv > 0:
            ax[0].text(j, i, "%.3f" % vv, color='white', ha='center', va='center')

for i, v in enumerate(b):
    for j, vv in enumerate(v):
        print(i, j, vv)
        if vv > 0:
            ax[1].text(j, i, "%.3f" % vv, color='black', ha='center', va='center')


plt.tight_layout()
plt.savefig("ngramplot.png")
plt.savefig("ngramplot.eps")
