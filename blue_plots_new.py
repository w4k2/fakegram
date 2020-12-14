import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

final_plots_scores = []

keys = ['text', 'author', 'title']
i_s = ['words', 'af']
quantities = np.array([.02, .05, .1, .25, .40])
GOLDEN = 1.61803399


blue = "#6666FF"
black = "#666666"
red = "#FF6666"

blue = "#5555DD"
black = "#555555"
red = "#DD5678"

bluec = "#AAAAFF"
blackc = "#AAAAAA"
redc = "#FFAAAA"

c_1 = [black, black, black]
ls_1 = [":", "--", "-"]

c_2 = [black, black, black, black]
ls_2 = [":", "--", "-.", "-"]


# Container for statistical analysis
stat_table = []

# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
scores = np.load("results/green.npy")
# KEYS x EXTRACTOR x REPEATS x FOLDS x QUANTITIES x FROM x TO
probas = np.load("results/green_probas.npy", allow_pickle=True)

# REPEATS x FOLDS x QUANTITIES -> REPEATS * FOLDS x QUANTITIES
y_true = np.load("results/green_ytest.npy", allow_pickle=True).reshape(10,5)
"""
Finding best n-gram range
"""
# KEYS x EXTRACTOR x FOLDS x QUANTITIES x FROM x TO
last_probas = probas.reshape(3,2,10,5,6,6)
# print(last_probas.shape)
# exit()
# KEYS x EXTRACTOR x QUANTITIES x FROM x TO
scores = np.mean(scores, axis=(2,3))

# N_TABLES x x REPEATS x FOLDS
# QUANTITIES x KEYS x EXTRACTOR
best_probas = np.zeros((5, 3, 2), dtype=object)

n_from = []
n_to = []
for q_id, quantity in enumerate(quantities):
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):

            green_score = scores[key_id, extractor_id, q_id]
            best_clf = np.argwhere(green_score==green_score.max())

            n_from.append(best_clf[0,0])
            n_to.append(best_clf[0,1])
            best_probas[q_id, key_id, extractor_id] = probas[key_id, extractor_id, :, :, q_id, best_clf[0,0], best_clf[0,1]]

# KEYS x EXTRACTOR
uni_scores = last_probas[:, :, :, 4, 0, 0]
bi_scores = last_probas[:, :, :, 4, 1, 1]
# Get unigrams and bigrams
last_probas_uni = []
last_probas_bi = []
last_probas_all = []
for extractor_id, i in enumerate(i_s):
    for key_id, key in enumerate(keys):
        uni = uni_scores[key_id, extractor_id]
        bi = bi_scores[key_id, extractor_id]
        last_probas_uni.append(uni)
        last_probas_bi.append(bi)
last_probas_all.append(last_probas_uni)
last_probas_all.append(last_probas_bi)
last_probas_all = np.array(last_probas_all)
# UNI/BI x METHOD x FOLDS
# print(last_probas_all.shape)

"""
Unigrams and bigrams plots by extractor
"""
# """
fig, ax = plt.subplots(1,2,figsize=(11, 4), sharey=True)
y_true = np.load("results/green_ytest.npy", allow_pickle=True).reshape(10,5)
last_scores = []
for type in range(last_probas_all.shape[0]):
    # METHOD x FOLDS
    type_probas = last_probas_all[type]
    words_probas = type_probas[:3, :]
    struct_probas = type_probas[3:, :]

    words_e = np.mean(words_probas, axis=0)
    struct_e = np.mean(struct_probas, axis=0)

    y_test = y_true[:, 4]
    for fold in range(10):
        pred_words_e = np.argmax(words_e[fold], axis=1)
        pred_struct_e = np.argmax(struct_e[fold], axis=1)
        last_scores.append(balanced_accuracy_score(y_test[fold], pred_words_e))
        last_scores.append(balanced_accuracy_score(y_test[fold], pred_struct_e))

last_scores = np.array(last_scores).reshape(4,10)
print(last_scores)
final_plots_scores = np.load("final_plots_scores.npy")
final_plots_scores = np.concatenate((final_plots_scores, last_scores), axis=0)
np.save("final_plots_scores", final_plots_scores)
print(final_plots_scores.shape)
exit()
"""
Plots by key
"""
"""
fig, ax = plt.subplots(1,3,figsize=(11, 3), sharey=True)
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
for key_id, key in enumerate(keys):
    print(key)
    plot_scores = []
    for q_id, quantity in enumerate(quantities):
        y_test = y_true[:, :, q_id].reshape(10,)
        separate_proba = best_probas[q_id, key_id, :]
        w = separate_proba[0].reshape(10,)
        s = separate_proba[1].reshape(10,)
        e = np.mean(best_probas[q_id, key_id, :], axis=0).reshape(10,)

        # calculate mean scores from folds
        fold_scores = np.zeros((3 ,10))
        for i in range(10):
            pred_w = np.argmax(w[i], axis=1)
            pred_s = np.argmax(s[i], axis=1)
            pred_e = np.argmax(e[i], axis=1)
            fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_w)
            fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_s)
            fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_e)
        plot_scores.append(np.mean(fold_scores, axis=1))
        if q_id == 4:
            stat_table.append(fold_scores)
    plot_scores = np.array(plot_scores).T

    # Lets make some plots
    labels = ['words', 'struct', 'ensemble']
    ls = [":", "--", "-"]
    for i in range(plot_scores.shape[0]):
        ax[key_id].plot(quantities, plot_scores[i], color=c_1[i], ls=ls_1[i], label=labels[i])


    ax[key_id].set_ylim(0.5, 1.0)
    ax[key_id].set_xlim(np.min(quantities), np.max(quantities))
    ax[key_id].set_xticks(quantities)#, [str(type) for type in quantities])
    ax[key_id].set_xticklabels(["%.0f%%" % (_*100) for _ in quantities])


    ax[key_id].set_title("%s" % key.upper(), fontsize=11, style='italic')
    ax[key_id].legend(frameon=False, ncol=1, loc=4, fontsize=12)
    ax[key_id].grid(ls=":", color=(0.85, 0.85, 0.85))


    ax[key_id].spines['right'].set_visible(False)
    ax[key_id].spines['top'].set_visible(False)


    ax[key_id].set_xlabel('quantity of used samples', fontsize=13)
    if key_id==0:
        ax[key_id].set_ylabel('balanced accuracy score', fontsize=13)

    plt.tight_layout()
plt.savefig("figures/features.png")
plt.savefig("figures/features.eps")
"""

"""
Plots by extractor
"""
"""
fig, ax = plt.subplots(1,2,figsize=(11, 4), sharey=True)
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
for extractor_id, extractor in enumerate(i_s):
    print(extractor)
    plot_scores = []
    for q_id, quantity in enumerate(quantities):
        y_test = y_true[:, :, q_id].reshape(10,)
        separate_proba = best_probas[q_id, :, extractor_id]

        txt = separate_proba[0].reshape(10,)
        aut = separate_proba[1].reshape(10,)
        ttl = separate_proba[2].reshape(10,)
        e = np.mean(best_probas[q_id, :, extractor_id], axis=0).reshape(10,)

        # calculate mean scores from folds
        fold_scores = np.zeros((4 ,10))
        for i in range(10):
            pred_txt = np.argmax(txt[i], axis=1)
            pred_aut = np.argmax(aut[i], axis=1)
            pred_ttl = np.argmax(ttl[i], axis=1)
            pred_e = np.argmax(e[i], axis=1)
            fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_txt)
            fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_aut)
            fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_ttl)
            fold_scores[3,i] = balanced_accuracy_score(y_test[i], pred_e)
        plot_scores.append(np.mean(fold_scores, axis=1))
        if q_id == 4:
            stat_table.append(fold_scores)
    plot_scores = np.array(plot_scores).T

    # Lets make some plots
    labels = ['text', 'author', 'title', 'ensemble']
    ls = [":", "--", "-.", "-"]

    for i in range(plot_scores.shape[0]):
        ax[extractor_id].plot(quantities, plot_scores[i], ls=ls_2[i], label=labels[i], color=c_2[i])


    ax[extractor_id].set_ylim(0.5, 1.0)
    ax[extractor_id].set_xlim(np.min(quantities), np.max(quantities))
    ax[extractor_id].set_xticks(quantities)#, [str(type) for type in quantities])
    ax[extractor_id].set_xticklabels(["%.0f%%" % (_*100) for _ in quantities])


    ax[extractor_id].set_title("%s" % extractor.upper(), fontsize=11, style='italic')
    ax[extractor_id].legend(frameon=False,ncol=2,loc=4,fontsize=12)
    ax[extractor_id].grid(ls=":", color=(0.85, 0.85, 0.85))


    ax[extractor_id].spines['right'].set_visible(False)
    ax[extractor_id].spines['top'].set_visible(False)


    ax[extractor_id].set_xlabel('quantity of used samples', fontsize=13)
    if extractor_id==0:
        ax[extractor_id].set_ylabel('balanced accuracy score', fontsize=13)

    plt.tight_layout()

    plt.tight_layout()

plt.savefig("figures/extractors.png")
plt.savefig("figures/extractors.eps")
"""
"""
Ensemble of 6
"""
y_true = np.load("results/green_ytest.npy", allow_pickle=True)
plot_scores = []
plot_stds = []
stds = []
for q_id, quantity in enumerate(quantities):
    y_test = y_true[:, :, q_id].reshape(10,)
    separate_proba = best_probas[q_id, :, :]

    txt_w = separate_proba[0,0].reshape(10,)
    aut_w = separate_proba[1,0].reshape(10,)
    ttl_w = separate_proba[2,0].reshape(10,)
    txt_s = separate_proba[0,1].reshape(10,)
    aut_s = separate_proba[1,1].reshape(10,)
    ttl_s = separate_proba[2,1].reshape(10,)
    e = np.mean(best_probas[q_id, :, :], axis=(0,1)).reshape(10,)
    e_w = np.mean(best_probas[q_id, :, 0], axis=0).reshape(10,)
    e_s = np.mean(best_probas[q_id, :, 1], axis=0).reshape(10,)

    # calculate mean scores from folds
    fold_scores = np.zeros((9 ,10))
    for i in range(10):
        pred_txt_w = np.argmax(txt_w[i], axis=1)
        pred_aut_w = np.argmax(aut_w[i], axis=1)
        pred_ttl_w = np.argmax(ttl_w[i], axis=1)
        pred_txt_s = np.argmax(txt_s[i], axis=1)
        pred_aut_s = np.argmax(aut_s[i], axis=1)
        pred_ttl_s = np.argmax(ttl_s[i], axis=1)
        pred_e = np.argmax(e[i], axis=1)
        pred_e_w = np.argmax(e_w[i], axis=1)
        pred_e_s = np.argmax(e_s[i], axis=1)
        fold_scores[0,i] = balanced_accuracy_score(y_test[i], pred_txt_w)
        fold_scores[1,i] = balanced_accuracy_score(y_test[i], pred_aut_w)
        fold_scores[2,i] = balanced_accuracy_score(y_test[i], pred_ttl_w)
        fold_scores[3,i] = balanced_accuracy_score(y_test[i], pred_txt_s)
        fold_scores[4,i] = balanced_accuracy_score(y_test[i], pred_aut_s)
        fold_scores[5,i] = balanced_accuracy_score(y_test[i], pred_ttl_s)
        fold_scores[8,i] = balanced_accuracy_score(y_test[i], pred_e)
        fold_scores[6,i] = balanced_accuracy_score(y_test[i], pred_e_w)
        fold_scores[7,i] = balanced_accuracy_score(y_test[i], pred_e_s)
    plot_scores.append(np.mean(fold_scores, axis=1))
    plot_stds.append(np.std(fold_scores, axis=1))
    if q_id == 4:
        stat_table.append(fold_scores)
        # WORDS, STRUCT, ALL
        final_plots_scores.append(fold_scores[6])
        final_plots_scores.append(fold_scores[7])
        final_plots_scores.append(fold_scores[8])
    stds.append(np.std(fold_scores, axis=1))

final_plots_scores = np.array(final_plots_scores)
np.save("final_plots_scores", final_plots_scores)

plot_scores = np.array(plot_scores).T
plot_stds = np.array(plot_stds).T
add = [0,3,1,4,2,5,6,7,8]
plot_scores = plot_scores[add]
plot_stds = plot_stds[add]
stds = np.array(stds).T

# Lets make some plots
labels = ['text', 'author', 'title','text', 'author', 'title', 'ensemble', 'ensemble', 'ensemble']

ls = [":", "--", "-.",":", "--", "-.", "-", "-", "-"]
colors = [bluec, bluec, bluec, redc, redc, redc, blue, red, black]
colorrs = [blue, blue, blue, red, red, red, blue, red, black]
colorrrs = [bluec, bluec, bluec, redc, redc, redc, bluec, redc, blackc]

lw = [1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]

labels = [labels[_] for _ in add]
ls = [ls[_] for _ in add]
colors = [colors[_] for _ in add]
colorrs = [colorrs[_] for _ in add]
colorrrs = [colorrrs[_] for _ in add]
lw = [lw[_] for _ in add]

#fig = plt.figure(figsize=(10, 5))
fig, ax = plt.subplots(1,1,figsize=(8, 4))
for i in range(plot_scores.shape[0]):
    ax.plot(quantities, plot_scores[i], color=colors[i], ls=ls[i], label=labels[i], lw=lw[i])

ax.set_ylim(0.5, 1.0)
ax.set_xlim(np.min(quantities), np.max(quantities))
ax.set_xticks(quantities)#, [str(type) for type in quantities])
ax.set_xticklabels(["%.0f%%" % (_*100) for _ in quantities])

# ax.set_title("%s" % extractor, fontsize=13, style='italic')
l = ax.legend(frameon=False,ncol=5,loc=4,fontsize=12)
for i, text in enumerate(l.get_texts()):
    text.set_color(colorrs[i])
ax.grid(ls=":", color=(0.85, 0.85, 0.85))


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax.set_xlabel('quantity of used samples', fontsize=13)
ax.set_ylabel('balanced accuracy score', fontsize=13)

plt.tight_layout()

plt.tight_layout()
plt.savefig("figures/ensemble.png")
plt.savefig("figures/ensemble.eps")

stat_table = np.array(stat_table)
np.save("results/blue_stat", stat_table)


print(plot_scores, plot_scores.shape)

bar_scores = plot_scores[:,-1]
bar_stds = plot_stds[:,-1]

fig, ax = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios':[2,1.5]}, figsize=(8,5))
print(bar_scores, bar_scores.shape)

d = .1
v = .05
print(labels)
a_bar = [0-d-v,0+d+v,1-d-v,1+d+v,2-d-v,2+d+v]
b_bar = [0,2*d,4*d]
ax[0].bar(a_bar, bar_scores[:6], align='center', width=d*2, color=colorrs[:6], zorder=100, ecolor=colorrrs[:6])
ax[0].set_xlim(-.5-d,2.5+d)

a = np.linspace(-1,5,10)

for i in range(2):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].grid(ls=":", color=(0.85, 0.85, 0.85), axis='y')

    ttt = np.linspace(.5,1,21)
    ax[i].set_yticks(ttt)
    ax[i].set_yticklabels([("%.2f" % _) if __%2==0 else ''
                           for __, _ in enumerate(ttt)], fontsize=10)

    for z in [-1, -2, -3]:
        dd = [bar_scores[z] - bar_stds[z] for _ in a]
        ee = [bar_scores[z] + bar_stds[z] for _ in a]
        ff = [bar_scores[z] for _ in a]
        #ax[i].fill_between(a, dd, ee,
        #                   color=colorrrs[z], zorder=0)
        #ax[i].plot(a, ff,color=colors[z], ls=':' if i==0 else '-', lw=.5)
        #ax[i].plot(a, ee,color=colors[z], ls="-", lw=.5)


ax[0].set_ylabel('balanced accuracy score', fontsize=13)
ax[0].set_title('feature models')

ax[1].set_title('ensembles')

ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels([_.upper() for _ in keys], fontsize=8)

ax[1].bar(b_bar, bar_scores[6:], align='center', width=d, color=colorrs[6:], ecolor=colorrrs[6:], zorder=100)

for i in range(6):
    ax[0].text(a_bar[i], bar_scores[i], ('%.3f' % bar_scores[i])[1:],
               ha='center', va='bottom', c = colorrs[i], fontsize=10)

for i in range(4):
    ax[1].text(b_bar[-i], bar_scores[-i], ('%.3f' % bar_scores[-i])[1:],
               ha='center', va='bottom', c = colorrs[-i], fontsize=10)


ax[1].set_xticks([0,2*d,4*d])
ax[1].set_xlim(0-.25,4*d+.25)
ax[1].set_xticklabels(['words'.upper(), 'af'.upper(), 'ensemble'.upper()], fontsize=8)



plt.ylim(.5,1)


plt.tight_layout()
plt.savefig("bar.png")
plt.savefig("figures/bar.eps")
