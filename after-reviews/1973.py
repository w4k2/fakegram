import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
# 3D
# fold | pattern | 2-klasy
datasets = {
    'old': ['text', 'title', 'author'],
    'new': ['text', 'title'],
}
n_splits = 2
n_repeats = 5

for dataset in datasets:

    y = np.load("all_y_test_%s.npy" % dataset)
    y = y.reshape(n_repeats, n_splits, y.shape[1])

    for key in datasets[dataset]:
        print("%s fold scores:" % key)
        fold_scores = []
        for repeat in range(n_repeats):
            y_repeat = y[repeat]

            proba_af = np.load("probas_bert/%i_%s_%s_struct.npy" % (repeat, key, dataset))
            proba_raw = np.load("probas_bert/%i_%s_%s.npy" % (repeat, key, dataset))

            for split in range(n_splits):
                print('# K:%s | F:%i:%i' % (key, repeat, split))

                yy = y_repeat[split]
                pp_af = proba_af[split,:,0]
                pp_raw = proba_raw[split,:,0]

                y_pred_af = pp_af < .5
                y_pred_raw = pp_raw < .5

                score_af = balanced_accuracy_score(yy, y_pred_af)
                score_raw = balanced_accuracy_score(yy, y_pred_raw)

                print("  AF %.3f\n RAW %.3f" % (
                    score_af,
                    score_raw
                ))

                """
                Unsure score
                """
                help_af = []
                help_raw = []
                q=10
                umar = np.linspace(0,.5,q)
                print("- HELP RAW WITH AF -")
                for unsure_margin in umar:
                    raw_unsure = (pp_raw > .5-unsure_margin) * (pp_raw < .5+unsure_margin)

                    y_pred_uns = np.copy(y_pred_raw)
                    y_pred_uns[raw_unsure] = y_pred_af[raw_unsure]

                    score_unsure = balanced_accuracy_score(yy, y_pred_uns)
                    help_af.append(score_unsure)

                    print("%.2f %.3f | %i samples" % (
                        unsure_margin, score_unsure, np.sum(raw_unsure)))

                print("- HELP AF WITH RAW -")
                for unsure_margin in umar:
                    af_unsure = (pp_af > .5 - unsure_margin) * (pp_af < .5+unsure_margin)

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

                ax[0].hist(pp_af, bins=32, alpha=.5, color='red')
                ax[0].hist(pp_raw, bins=32, alpha=.5, color='blue')

                ax[0].set_title('AF support distribution [%s | K:%s | F:%i:%i]' % (dataset, key, repeat, split))

                ax[1].plot(umar, help_af, c='red', label='AF-help')
                ax[1].plot(umar, help_raw, c='blue', label='RAW-help')
                ax[1].hlines(score_af, 0, .5, color='red', ls=":", label='AF')
                ax[1].hlines(score_raw, 0, .5, color='blue', ls=":", label='RAW')

                ax[1].legend()

                plt.tight_layout()
                plt.savefig('figures/%s_%s_%i_%i.png' % (dataset, key, repeat, split))
                plt.savefig('foo.png')

                #exit()
