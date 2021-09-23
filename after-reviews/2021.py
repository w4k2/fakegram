import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

datasets = {
    'old': ['text', 'title', 'author'],
    'new': ['text', 'title'],
}
n_splits = 2
n_repeats = 5

q=15
umar = np.linspace(0,.5,q)

for dataset in datasets:
    y = np.load("all_y_test_%s.npy" % dataset)
    y = y.reshape(n_repeats, n_splits, y.shape[1])

    probas_af = []
    probas_raw = []

    for key in datasets[dataset]:
        aaa_af = []
        aaa_raw = []
        for repeat in range(n_repeats):
            proba_af = np.load("probas_bert/%i_%s_%s_struct.npy" % (repeat, key, dataset))
            proba_raw = np.load("probas_bert/%i_%s_%s.npy" % (repeat, key, dataset))

            aaa_af.append(proba_af)
            aaa_raw.append(proba_raw)

            #for split in range(n_splits):
            #    print('# %s:K:%s:F:%i:%i' % (dataset, key, repeat, split))

        probas_af.append(aaa_af)
        probas_raw.append(aaa_raw)

    # key x repeat x fold x pattern x class
    probas_af = np.array(probas_af)
    probas_raw = np.array(probas_raw)

    # repeat x fold x pattern
    probas_af = np.mean(probas_af, axis=0)[:,:,:,0]
    probas_raw = np.mean(probas_raw, axis=0)[:,:,:,0]

    # STORAGE
    scores_af = []
    scores_raw = []
    helps_af = []
    helps_raw = []

    for repeat in range(n_repeats):
        for split in range(n_splits):
            # Gather data
            af = probas_af[repeat, split]
            raw = probas_raw[repeat, split]
            yy = y[repeat, split]

            # Establish base prediction
            y_pred_af = af < .5
            y_pred_raw = raw < .5

            # Establish base scores
            score_af = balanced_accuracy_score(yy, y_pred_af)
            score_raw = balanced_accuracy_score(yy, y_pred_raw)

            print(dataset, repeat, split, yy.shape, af.shape, raw.shape)

            print("  AF %.3f" % score_af)
            print(" RAW %.3f" % score_raw)

            """
            Unsure score
            """
            help_af = []
            help_raw = []
            print("- HELP RAW WITH AF -")
            for unsure_margin in umar:
                raw_unsure = (raw > .5-unsure_margin) * (raw < .5+unsure_margin)

                y_pred_uns = np.copy(y_pred_raw)
                y_pred_uns[raw_unsure] = y_pred_af[raw_unsure]

                score_unsure = balanced_accuracy_score(yy, y_pred_uns)
                help_af.append(score_unsure)

                print("%.2f %.3f | %i samples" % (
                    unsure_margin, score_unsure, np.sum(raw_unsure)))

            print("- HELP AF WITH RAW -")
            for unsure_margin in umar:
                af_unsure = (af > .5 - unsure_margin) * (af < .5+unsure_margin)

                y_pred_uns = np.copy(y_pred_af)
                y_pred_uns[af_unsure] = y_pred_raw[af_unsure]

                score_unsure = balanced_accuracy_score(yy, y_pred_uns)
                help_raw.append(score_unsure)

                print("%.2f %.3f | %i samples" % (
                    unsure_margin, score_unsure, np.sum(af_unsure)))

            """
            PLOT
            """
            fig, ax = plt.subplots(1,2,figsize=(10,5))

            ax[0].hist(af, bins=32, alpha=.5, color='red', label='AF')
            ax[0].hist(raw, bins=32, alpha=.5, color='blue', label='WORDS')

            ax[1].plot(umar, help_raw, c='red', label='AF-resolved WORDS')
            ax[1].plot(umar, help_af, c='blue', label='WORDS-resolved AF')
            ax[1].hlines(score_af, 0, .5, color='red', ls=":", label='AF')
            ax[1].hlines(score_raw, 0, .5, color='blue', ls=":", label='RAW')

            """
            Stylize
            """
            ax[0].set_title('Positive class support distribution [%s]' % ('ensemble'))
            ax[1].set_title('Serial ensemble')

            ax[0].legend(frameon=False)
            legend = ax[1].legend(ncol=2, loc=8)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('white')


            for a in ax:
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.grid(ls=":")

            ax[0].set_xlabel('Positive class support')
            ax[1].set_xlabel('Unsure margin')
            ax[0].set_ylabel('Quantity of samples')
            ax[1].set_ylabel('Accuracy')

            plt.tight_layout()
            plt.savefig('figures/%s_%s_%i_%i.png' % (dataset, 'ensemble', repeat, split))
            plt.savefig('foo.png')

            exit()

            # STORE
            scores_af.append(score_af)
            scores_raw.append(score_raw)
            helps_af.append(help_af)
            helps_raw.append(help_raw)

    scores_af = np.array(scores_af)
    scores_raw = np.array(scores_raw)
    helps_af = np.array(helps_af)
    helps_raw = np.array(helps_raw)


    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ax.set_title('[%s | %s]' % (dataset, 'ensemble'))

    aaaa = np.mean(helps_af, axis=0)
    bbbb = np.mean(helps_raw, axis=0)
    cccc = np.mean(scores_af, axis=0)
    dddd = np.mean(scores_raw, axis=0)

    ax.plot(umar, aaaa, c='red', label='AF-help [%.3f]' % np.max(aaaa))
    ax.plot(umar, bbbb, c='blue', label='RAW-help [%.3f]' % np.max(bbbb))
    ax.hlines(cccc, 0, .5, color='red', ls=":", label='AF [%.3f]' % cccc)
    ax.hlines(dddd, 0, .5, color='blue', ls=":", label='RAW [%.3f]' % dddd)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig('figures/%s_%s.png' % (dataset, 'ensemble'))
    plt.savefig('foo.png')
