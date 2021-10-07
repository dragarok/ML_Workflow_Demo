#!/usr/bin/env python3


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SVMSMOTE

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.callbacks import Callback
import pandas as pd
import numpy as np
import datetime
import os
import yaml
import tf2onnx
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dvclive.keras import DvcLiveCallback
from sklearn.utils.class_weight import compute_class_weight


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average="weighted")
        _val_recall = recall_score(val_targ, val_predict, average="weighted")
        _val_precision = precision_score(val_targ, val_predict, average="weighted")

        logs["val_f1"] = _val_f1
        logs["val_recall"] = _val_recall
        logs["val_precision"] = _val_precision
        print(
            " — val_f1: %f — val_precision: %f — val_recall: %f"
            % (_val_f1, _val_precision, _val_recall)
        )
        return


def run_model_training():
    """Run model training using tensorflow

    # TODO Refactor the function
    Args:

    Output:
        None"""

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    MODEL_NAME = "keras_model"
    SEED = params["seed"]
    GCP_BUCKET = params["gcp_bucket"]
    TEST_SIZE = params["test_size"]

    EPOCHS = params["train"]["epochs"]
    BATCH_SIZE = params["train"]["batch_size"]
    ACTIVATION = params["train"]["activation"]
    LAYERS = params["train"]["fc_layers"]
    LEARNING_RATE = params["train"]["learning_rate"]

    EVAL_BATCH_SIZE = params["eval"]["batch_size"]

    df = pd.read_csv("Reduced_Features.csv")
    features_df = df.drop(["Label"], axis=1)
    labels_df = df["Label"]

    N_FEATURES = len(features_df.columns)
    # Since labels are from 0 to max value
    N_LABELS = max(labels_df.unique()) + 1
    N_LABELS = N_LABELS.item()

    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(features_df, labels_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED)

    class_labels = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
    class_weight = dict(zip(class_labels, class_weights))
    for i in range(N_LABELS):
        if i not in class_weight.keys():
            class_weight[i] = 1
    
    print(class_weight)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_data = val_data.batch(EVAL_BATCH_SIZE)
    # TODO Optimization using prefetch

    fc_layers = []
    for x in LAYERS:
        fc_layers.append(tf.keras.layers.Dense(x, activation=ACTIVATION))
        fc_layers.append(tf.keras.layers.Dropout(rate=0.3))

    model = tf.keras.Sequential(
        fc_layers + [tf.keras.layers.Dense(N_LABELS, activation="softmax")]
    )

    checkpoint_path = os.path.join("feat-sel-check", "save_at_{epoch}")
    tensorboard_path = "logs"

    callbacks = [
        # TensorBoard will store logs for each epoch and graph performance for us.
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        # ModelCheckpoint will save models after each epoch for retrieval later.
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            mode="max",
        ),
        # EarlyStopping will terminate training when val_loss ceases to improve.
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        DvcLiveCallback(model_file="saved_model.h5"),
        # TODO Looks like it fixes the metric we have been looking for
        Metrics(valid_data=(X_test, y_test)),
    ]

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5))
    # TODO Test with more data for F1score as well as fbeta score
    # metrics=[tfa.metrics.FBetaScore(num_classes=N_LABELS, average="micro", threshold=0.9), f1])

    # model.fit(train_data, callbacks=callbacks, epochs=1)
    history = model.fit(
        train_data, epochs=EPOCHS, validation_data=val_data, callbacks=callbacks,)
    print("Model training done")
    print(history.history)
    print("\n\n\n")

    losses_df = pd.DataFrame(
        columns=["epoch", "loss", "val_loss", "val_f1", "val_precision", "val_recall"]
    )
    for i in range(EPOCHS):
        losses_df = losses_df.append(
            {
                "epoch": i,
                "loss": history.history["loss"][i],
                "val_loss": history.history["val_loss"][i],
                "val_f1": history.history["val_f1"][i],
                "val_precision": history.history["val_precision"][i],
                "val_recall": history.history["val_recall"][i],
            },
            ignore_index=True,
        )
    losses_df.to_csv("losses.csv", index=False)
    print("Saved losses to losses file")

    # Confusion Matrix to be plotted by dvc
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred))
    confusion_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    confusion_df.to_csv("classes.csv", index=False)

    # Working on metrics
    metrics = {}
    metrics["train"] = {}
    metrics["train"]["loss"] = min(history.history["loss"])
    metrics["train"]["final_loss"] = history.history["loss"][-1]
    metrics["eval"] = {}
    metrics["eval"]["loss"] = min(history.history["val_loss"])
    metrics["eval"]["final_loss"] = history.history["val_loss"][-1]
    metrics["eval"]["final_f1"] = history.history["val_f1"][-1]
    with open("metrics.json", "w") as outfile:
        json.dump(metrics, outfile)
    print(metrics)

    # Convert model to onnx
    spec = (tf.TensorSpec((None, N_FEATURES), tf.float32, name="input"),)
    output_path = MODEL_NAME + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(
        model, opset=13, output_path=output_path, input_signature=spec
    )
    print("Converted and saved the model to ONNX file")


if __name__ == "__main__":
    run_model_training()
