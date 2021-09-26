#!/usr/bin/env python3

import tensorflow as tf
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os

def run_model_training():
    """Run model training using tensorflow

    # TODO Refactor the function
    Args:

    Output:
        None"""
    gcp_bucket = 'tfc-cml'
    BATCH_SIZE = 32

    df = pd.read_csv("../1_Label_And_Feature_Workflow/Full_Features.csv")

    features_df = df.drop(['Label', 'Bar'], axis=1)
    labels_df = df['Label']

    sel_features_df = pd.read_csv('Selected_Features.csv')
    sel_features_df = sel_features_df.sort_values(by='Sklearn_Rank')
    k_best_cols = list(sel_features_df[:300]['Col_Name'])
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

    # model.fit(train_dataset, callbacks=callbacks, epochs=1)
    model.fit(train_dataset, epochs=1)

    MODEL_PATH = "../2_Training_Workflow/keras-model"
    # SAVE_PATH = os.path.join("gs://", gcp_bucket, MODEL_PATH)
    SAVE_PATH = MODEL_PATH
    model.save(SAVE_PATH)

    model = tf.keras.models.load_model(SAVE_PATH)

    print(model.evaluate(test_dataset, batch_size=32))
    print("Done evaluating the model")
    return MODEL_PATH

if __name__ == "__main__":
    run_model_training()