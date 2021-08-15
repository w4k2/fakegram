from gensim.models import FastText
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from nltk.corpus import stopwords
import re

# Parameters
keys = ['title']
n_splits = 2
n_repeats = 5
# % of data used
quantity = .01
random_state = 1410
eng_stopwords = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return words

def convert_to_vector(text):
    words = clean_text(text)
    array = np.asarray([ft.wv[w] for w in words if w in ft.wv],dtype='float32')
    return array.mean(axis=0)

#df = pd.read_csv("../data/data.csv")
df = pd.read_csv("data/data_with_labels.csv")
y = df['label'].values.astype(int)

rskf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

for key_id, key in enumerate(keys):
    print("### KEY " + key)
    #df[key] = ['__label__%i %s' % (int(y[s_id]), s) for s_id, s in enumerate(df[key])]
    X = df[key].to_numpy().astype('U')

    for repeat in range(n_repeats):
        X_s, y_s = resample(X, y, n_samples=6400, replace=False, stratify=y, random_state=random_state+repeat)

        probas = []
        for fold_id, (train,test) in enumerate(rskf.split(X_s, y_s)):
            X_train, X_test = X_s[train], X_s[test]
            y_train, y_test = y_s[train], y_s[test]

            X_train_ft = [clean_text(s) for s in X_train]
            ft = FastText(X_train_ft, vector_size = 4, window = 4, min_count = 1)

            filename = 'ft_model_%s_%i_%i' % (key, repeat, fold_id)
            ft.save('ft_models/' + filename + '.model')
            ft.wv.save_word2vec_format('ft_models/' + filename + '.txt', binary = False)

            print(ft.wv.most_similar('cat'))

            train_data_fasttext = [convert_to_vector(text) for text in X_train]
            mlp = MLPClassifier(random_state=random_state)
            mlp = mlp.fit(train_data_fasttext, y_train)
            X_test = X_test.reshape(-1, 1)
            y_pred = mlp.predict(X_test)
            print(balanced_accuracy_score(y_test, y_pred))
