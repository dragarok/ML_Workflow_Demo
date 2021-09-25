#!/usr/bin/env python3

import tensorflow as tf
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os

gcp_bucket = 'tfc-cml'
BATCH_SIZE = 32

df = pd.read_csv("Full_Features.csv")

features_df = df.drop(['Label', 'Bar'], axis=1)
labels_df = df['Label']

k_best_cols_idx = (
    SelectKBest(chi2, k=300)
    .fit(features_df, labels_df)).get_support(indices=True)
f_cols_list = list(features_df.columns)
k_best_cols = [f_cols_list[i] for i in k_best_cols_idx]

k_best_features_df = features_df[k_best_cols]

X_train, X_test, y_train, y_test = train_test_split(k_best_features_df, labels_df,
                                                    test_size=0.2,
                                                    random_state=42)
# FIXME Error with dataset sharding
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_data.batch(BATCH_SIZE)
test_dataset = test_data.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(14, activation='softmax')
])

checkpoint_path = os.path.join("gs://", gcp_bucket, "feat-sel-check", "save_at_{epoch}")

tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
    "gs://", gcp_bucket, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

callbacks = [
    # TensorBoard will store logs for each epoch and graph performance for us.
    # tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    # ModelCheckpoint will save models after each epoch for retrieval later.
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
    # EarlyStopping will terminate training when val_loss ceases to improve.
    # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_dataset, callbacks=callbacks, epochs=1)

MODEL_PATH = "keras-feat-sel"
SAVE_PATH = os.path.join("gs://", gcp_bucket, MODEL_PATH)
model.save(SAVE_PATH)

model = tf.keras.models.load_model(SAVE_PATH)

print(model.evaluate(test_dataset, batch_size=32))

print("Done fitting the model")
