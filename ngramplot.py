import numpy as np
import matplotlib.pyplot as plt

data = np.mean(np.load("n_gram_scores.npy"), axis=0)

a = data[1]
b = data[0]
c = data[0]
# FOLDS x METHOD x I x J
print(data[0], data.shape)

fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))

ax[0].imshow(a, cmap="Reds", vmax=1, vmin=.5, origin='lower')
ax[1].imshow(b, cmap="Greens", vmax=1, vmin=.5, origin='lower')
ax[2].imshow(c, cmap="Blues", vmax=1, vmin=.5, origin='lower')
ax[0].set_title("Words")
ax[1].set_title("Alphabet Flatting")
ax[2].set_title("Case sensitive Alphabet Flatting")

for z in range(3):
    ax[z].set_xticks(range(7))
    ax[z].set_xticklabels(np.array(list(range(7)))+1)
    ax[z].set_yticks(range(7))
    ax[z].set_yticklabels(np.array(list(range(7)))+1)
    ax[z].set_ylabel("from")
    ax[z].set_xlabel("to")

for i, v in enumerate(a):
    for j, vv in enumerate(v):
        print(i, j, vv)
        if vv > 0:
            ax[0].text(j, i, "%.3f" % vv, color='black',
                       ha='center', va='center')

for i, v in enumerate(b):
    for j, vv in enumerate(v):
        print(i, j, vv)
        if vv > 0:
            ax[1].text(j, i, "%.3f" % vv, color='white',
                       ha='center', va='center')

for i, v in enumerate(c):
    for j, vv in enumerate(v):
        print(i, j, vv)
        if vv > 0:
            ax[2].text(j, i, "%.3f" % vv, color='white',
                       ha='center', va='center')


plt.tight_layout()
plt.savefig("ngramplot.png")
plt.savefig("ngramplot.eps")
