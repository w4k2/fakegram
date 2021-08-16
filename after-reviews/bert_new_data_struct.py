"""
ADDITIONAL DEPENDENCIES:
pip install tensorflow-hub
pip install tensorflow-text
pip install tf-models-official
"""
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from bert_utils import handle, preprocess
from official.nlp import optimization
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from strlearn.metrics import balanced_accuracy_score


"""
BERT definition
"""
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
tfhub_handle_encoder = handle(bert_model_name)
tfhub_handle_preprocess = preprocess(bert_model_name)

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(net)
  return tf.keras.Model(text_input, net)

"""
Experiment
"""
# Parameters
keys = ['text', 'title']
n_splits = 2
n_repeats = 5
# % of data used
quantity = .01
random_state = 1410

# Load CSV
df_words = pd.read_csv('data/data_with_labels_struct.csv')
# Renaming 'article' to 'text'
df_words.columns = ['title', 'text', 'label']
y = df_words['label'].values.astype(int)

# CV
rskf = StratifiedKFold(
    n_splits=n_splits, random_state=random_state, shuffle=True)

for key_id, key in enumerate(keys):
    # U because of nan
    X = df_words[key].to_numpy().astype('U')

    for repeat in range(n_repeats):
        X_s, y_s = resample(X, y, n_samples=6400, replace=False, stratify=y, random_state=random_state + repeat)

        # Probas container
        probas = []
        for fold_id, (train, test) in enumerate(rskf.split(X_s, y_s)):
            print(key, repeat, fold_id)
            X_train, X_test = X_s[train], X_s[test]
            y_train, y_test = y_s[train], y_s[test]

            # t to categorical
            y_train_c = tf.keras.utils.to_categorical(y_train)
            y_test_c = tf.keras.utils.to_categorical(y_test)

            # BERT model
            clf = build_classifier_model()
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            metrics = tf.metrics.CategoricalAccuracy()
            epochs = 5
            steps_per_epoch = np.sqrt(X.shape[0])
            num_train_steps = steps_per_epoch * epochs
            num_warmup_steps = int(0.1*num_train_steps)

            init_lr = 3e-5
            optimizer = optimization.create_optimizer(
                init_lr=init_lr,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                optimizer_type='adamw'
            )

            clf.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            history = clf.fit(
                x=X_train,
                y = y_train_c,
                # validation_data=None,
                epochs=epochs
            )

            proba = clf.predict(X_test)
            # pred = np.argmax(proba, axis=1)
            # score = balanced_accuracy_score(y_test, pred)
            # print(score)
            # exit()
            probas.append(proba)
        probas = np.array(probas)
        np.save("probas_bert/%i_%s_new_struct" % (repeat, key), probas)
        # print(probas.shape)
        # exit()
