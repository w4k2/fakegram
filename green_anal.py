import numpy as np
import matplotlib.pyplot as plt

keys = ['text', 'author', 'title']
i_s = ['words', 'af']
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
    if q_id != 4:
        continue
    fig, ax = plt.subplots(2,3,figsize=(9.5,7),sharex=True,sharey=True)
    for key_id, key in enumerate(keys):
        for extractor_id, i in enumerate(i_s):
            print(q_id, key_id, extractor_id)

            green_score = scores[key_id, extractor_id, q_id]

            print(green_score.shape)

            #plt.clf()
            #fig = plt.figure(figsize=(4,4))

            ppp = ax[extractor_id, key_id]


            #ppp.yaxis.tick_right()
            #ppp.yaxis.set_label_position("right")

            ppp.imshow(np.fliplr(green_score), cmap="Greys", vmin=.5, vmax=1, origin='lower')
            plt.xticks(range(n_range), ranger)
            plt.yticks(range(n_range), ranger)
            if key_id == 0:
                ppp.set_ylabel("from")
            if extractor_id == 1:
                ppp.set_xlabel("to")


            ppp.spines['right'].set_visible(False)
            ppp.spines['top'].set_visible(False)
            ppp.spines['left'].set_visible(False)
            ppp.spines['bottom'].set_visible(False)

            white = np.array([1,1,1])
            for start in range(n_range):
                for end in range(n_range):
                    if start >= end:
                        val = green_score[end,start]
                        ppp.text(5-start, end, ("%.3f" % val)[1:],
                                 ha='center', weight='bold', va='center', c=white if val>.75 else white*0,fontsize=10)

            ppp.set_title("%s on %s preprocessing" % (key.upper(), i.upper()), fontsize=8)

            """
            filename = "k_%s_q_%s_e_%s" % (key, ("%.2f" % quantity)[2:], i)
            plt.title("%s on %s extractor" % (key, i), fontsize=10)

            """
            plt.tight_layout()
            plt.savefig("foo.png")
            #plt.savefig("figures/%s.png" % filename)
            #plt.savefig("figures/%s.eps" % filename)
        plt.savefig("figures/heat.eps")
