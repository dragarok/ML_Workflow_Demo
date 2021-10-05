#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
import optuna
from optuna.integration.tfkeras import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances

BATCHSIZE = 128
CLASSES = 10
EPOCHS = 30
N_TRAIN_EXAMPLES = 3000
STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCHSIZE / 10)
VALIDATION_STEPS = 30

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

        _val_f1 = f1_score(val_targ, val_predict, average='weighted', zero_division=0)
        _val_recall = recall_score(val_targ, val_predict, average='weighted', zero_division=0)
        _val_precision = precision_score(val_targ, val_predict, average='weighted', zero_division=0)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


def create_model(trial):
    """ Optuna create model with hyperparameters """

    # Hyperparameters to be tuned by Optuna.
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    FIRST_LAYER = trial.suggest_categorical("first_units", [128, 256, 512])
    SECOND_LAYER = trial.suggest_categorical("second_units", [64, 128, 256])
    THIRD_LAYER = trial.suggest_categorical("third_units", [32, 64, 128, 256])
    DROPOUT = trial.suggest_discrete_uniform("dropout_rate", 0.1, 0.5, 0.1)

    # Compose neural network with one hidden layer.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=FIRST_LAYER, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=DROPOUT))
    model.add(tf.keras.layers.Dense(units=SECOND_LAYER, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=DROPOUT))
    model.add(tf.keras.layers.Dense(units=THIRD_LAYER, activation='relu'))
    model.add(tf.keras.layers.Dense(14, activation='softmax'))

    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        # metrics=["accuracy"],
    )
    return model


def objective(trial):
    """Create an objective for optuna"""

    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Parameters
    EVAL_BATCH_SIZE = 64
    TEST_SIZE = 0.2
    SEED = 42
    BATCH_SIZE = 64

    # Metrics to be monitored by Optuna.
    monitor = "val_f1"

    df = pd.read_csv("Reduced_Features.csv")
    features_df = df.drop(['Label'], axis=1)
    labels_df = df['Label']

    N_FEATURES = len(features_df.columns)
    # Since labels are from 0 to max value
    N_LABELS = max(labels_df.unique()) + 1
    N_LABELS = N_LABELS.item()

    X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df,
                                                        test_size=TEST_SIZE,
                                                        random_state=SEED)


    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_data = val_data.batch(EVAL_BATCH_SIZE)

    model = create_model(trial)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        Metrics(valid_data=(X_test, y_test)),
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )

    return history.history[monitor][-1]


def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )

    study.optimize(objective, n_trials=15, timeout=800)

    show_result(study)

    plot_parallel_coordinate(study)
    plot_param_importances(study)

    # run_model_training()
