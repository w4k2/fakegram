import pandas as pd

full_sources_labels = pd.read_csv("data/full_sources_labels.csv")
sources_labels_nan = full_sources_labels[['source','NewsGuard, overall_class']]
sources_labels = sources_labels_nan.dropna()

sources_labels = sources_labels.rename(columns={'NewsGuard, overall_class': 'label'})
sources_labels.set_index('source',inplace=True)
sources_labels = sources_labels.T

data = pd.read_csv("data/joined_data.csv")

articles = []
titles = []
data_labels = []
good_articles = 0
bad_articles = 0
thereshold = 10000
for index, row in data.iterrows():
    try:
        current_label = sources_labels[row['sources']].label.item()
        if (good_articles < thereshold and current_label == 1):
            data_labels.append(int(current_label))
            articles.append(row['articles'])
            titles.append(row['titles'])
            good_articles = good_articles + 1
        elif (bad_articles < thereshold and current_label == 0):
            data_labels.append(int(current_label))
            articles.append(row['articles'])
            titles.append(row['titles'])
            bad_articles = bad_articles + 1
    except KeyError as exp:
        data.drop(index, inplace=True)
    if (good_articles == thereshold and bad_articles == thereshold):
        break

dictionary = {}
dictionary['title'] = titles
dictionary['article'] = articles
dictionary['label'] = data_labels

data = pd.DataFrame(data=dictionary)

data.to_csv('data/data_with_labels.csv', index=False)
