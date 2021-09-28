#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import datetime
import os
import yaml
import tf2onnx
import matplotlib.pyplot as plt
import seaborn as sns

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

    N_FEATURES = len(features_df.columns)
    # Since labels are from 0 to max value
    N_LABELS = max(labels_df.unique()) + 1
    print(type(N_LABELS))
    print(type(N_LABELS.item()))
    print(N_LABELS)

    MODEL_NAME = 'keras_model'

    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df,
                                                        test_size=TEST_SIZE,
                                                        random_state=SEED)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_data = val_data.batch(EVAL_BATCH_SIZE)
    # TODO Optimization using prefetch

    fc_layers = []
    for x in LAYERS:
        fc_layers.append(tf.keras.layers.Dense(x, activation=ACTIVATION))

    model = tf.keras.Sequential(
        fc_layers + [tf.keras.layers.Dense(N_LABELS, activation='softmax')]
    )

    checkpoint_path = os.path.join("gs://", GCP_BUCKET, "feat-sel-check", "save_at_{epoch}")

    # tensorboard_path = os.path.join(  # Timestamp included to enable timeseries graphs
    #     "gs://", GCP_BUCKET, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # )
    tensorboard_path = "logs"

    callbacks = [
        # TensorBoard will store logs for each epoch and graph performance for us.
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        # ModelCheckpoint will save models after each epoch for retrieval later.
        # tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        # EarlyStopping will terminate training when val_loss ceases to improve.
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  # TODO More look on what the metric should be
                  metrics=[tfa.metrics.FBetaScore(num_classes=N_LABELS, average="micro", threshold=0.9)])

    # model.fit(train_data, callbacks=callbacks, epochs=1)
    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=callbacks)
    print("Model training done")

    # Make a plot of validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_plot.png')
    plt.close()

    # Confusion Matrix to be plotted by dvc
    y_pred = np.argmax(model.predict(X_test), axis=1)
    confusion_df = pd.DataFrame({'actual': y_test,
                                  'predicted': y_pred})
    confusion_df.to_csv("classes.csv", index=False)


    # Plot confusion matrix using seaborn
    conf_matrix = confusion_matrix(y_test, y_pred)
    label_classes = sorted(labels_df.unique())
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(label_classes)
    ax.yaxis.set_ticklabels(label_classes)
    plt.savefig('confusion.png')

    # Working on metrics
    metrics = {}
    metrics["train"] = {}
    metrics["train"]["loss"] = max(history.history['loss'])
    metrics["eval"] = {}
    # metrics["eval"]["epochs"] = np.argmax(history.history['val_custom_f1'])
    # metrics["eval"]["f1_score"] = max(history.history['val_custom_f1'])
    print(metrics)

    # Convert model to onnx
    # spec = (tf.TensorSpec((None, N_FEATURES), tf.float32, name="input"),)
    output_path = MODEL_NAME + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=output_path)
    print("Converted and saved the model to ONNX file")

if __name__ == "__main__":
    run_model_training()
