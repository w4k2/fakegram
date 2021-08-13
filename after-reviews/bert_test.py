import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
from bert_utils import handle, preprocess
from official.nlp import optimization
from strlearn.metrics import balanced_accuracy_score


df = pd.read_csv('data/data.csv')
# U because of nan
X_title = df["title"].to_numpy().astype('U')
y_n = df["label"]

y = tf.keras.utils.to_categorical(y_n)

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

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = tf.metrics.CategoricalAccuracy()
epochs = 5
steps_per_epoch = 200
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model = build_classifier_model()
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=X_title[:200],
                               y = y[:200],
                               validation_data=(X_title[200:400], y[200:400]),
                               epochs=epochs)
print(X_title[400:600])
proba = classifier_model.predict(X_title[400:600])
print(proba)
pred = np.argmax(proba, axis=1)
score = balanced_accuracy_score(y_n[400:600], pred)
print(score)
