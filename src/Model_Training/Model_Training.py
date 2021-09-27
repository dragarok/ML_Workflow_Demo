#!/usr/bin/env python3

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
import yaml

def run_model_training():
    """  Run model training using tensorflow

    # TODO Refactor the function
    Args:

    Output:
        None"""

    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    SEED = params['seed']
    GCP_BUCKET = params['gcp_bucket']
    TEST_SIZE = params['test_size']

    EPOCHS = params['train']['epochs']
    BATCH_SIZE = params['train']['batch_size']
    ACTIVATION = params['train']['activation']
    LAYERS = params['train']['fc_layers']

    EVAL_BATCH_SIZE = params['eval']['batch_size']

    df = pd.read_csv("Reduced_Features.csv")

    features_df = df.drop(['Label'], axis=1)
    labels_df = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df,
                                                        test_size=TEST_SIZE,
                                                        random_state=SEED)
    # FIXME Error with dataset sharding
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_data.batch(BATCH_SIZE)
    test_dataset = test_data.batch(BATCH_SIZE)

    fc_layers = []
    for x in LAYERS:
        fc_layers.append(tf.keras.layers.Dense(x, activation=ACTIVATION))

    model = tf.keras.Sequential(
        fc_layers + [tf.keras.layers.Dense(14, activation='softmax')]
    )

    checkpoint_path = os.path.join("gs://", GCP_BUCKET, "feat-sel-check", "save_at_{epoch}")

    tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
        "gs://", GCP_BUCKET, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    model.fit(train_dataset, epochs=EPOCHS)

    MODEL_PATH = "../2_Training_Workflow/keras-model"
    # SAVE_PATH = os.path.join("gs://", gcp_bucket, MODEL_PATH)
    SAVE_PATH = MODEL_PATH
    model.save(SAVE_PATH)

    model = tf.keras.models.load_model(SAVE_PATH)

    print(model.evaluate(test_dataset, batch_size=EVAL_BATCH_SIZE))
    print("Done evaluating the model")
    return MODEL_PATH

if __name__ == "__main__":
    run_model_training()
