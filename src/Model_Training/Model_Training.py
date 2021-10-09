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
import nvgpu
from optuna.integration.tfkeras import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from imblearn.over_sampling import SMOTE



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
    """ Create model using individual dense layers"""

    # Read hyperparams from params file dvc
    ACTIVATION = core_params["activation"]
    N_LAYERS = core_params["n_layers"]
    LEARNING_RATE = core_params["learning_rate"]
    DROPOUT_RATE = core_params["dropout"]
    HIDDEN_UNITS = core_params["hidden_units"]
    N_LABELS = 14

    # Create trials for optuna
    ACTIVATION = trial.suggest_categorical("activation", ACTIVATION)
    N_LAYERS = trial.suggest_int('n_layers', 1, N_LAYERS)
    LEARNING_RATE = trial.suggest_float("learning_rate", LEARNING_RATE[0], LEARNING_RATE[1], log=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(N_LAYERS):
        NUM_HIDDEN = trial.suggest_categorical("n_units_l{}".format(i), HIDDEN_UNITS)
        model.add(tf.keras.layers.Dense(NUM_HIDDEN, activation=ACTIVATION))
        DROPOUT = trial.suggest_discrete_uniform("dropout_l{}".format(i), DROPOUT_RATE[0], DROPOUT_RATE[1], DROPOUT_RATE[2])
        model.add(tf.keras.layers.Dropout(rate=DROPOUT))
    model.add(tf.keras.layers.Dense(N_LABELS, activation='softmax'))

    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
    )
    return model


def objective(trial, return_model=False):
    """Create an objective for optuna"""

    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Parameters
    EPOCHS = params["epochs"]
    BATCH_SIZE = params["batch_size"]
    OPTIMIZER = core_params['optimizer']

    # Hyperparameters to be tuned by Optuna.
    # EPOCHS = trial.suggest_categorical("epochs", EPOCHS)
    # BATCH_SIZE = trial.suggest_categorical("batch_size", BATCH_SIZE)
    N_TRAIN_EXAMPLES = 3000
    VALIDATION_STEPS = 30
    STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCH_SIZE / 10)

    # Metrics to be monitored by Optuna.
    monitor = "val_f1"

    df = pd.read_csv("Reduced_Features.csv")
    features_df = df.drop(['Label'], axis=1)
    labels_df = df['Label']

    N_FEATURES = len(features_df.columns)
    # Since labels are from 0 to max value
    N_LABELS = max(labels_df.unique()) + 1
    N_LABELS = N_LABELS.item()

    oversample = SMOTE()
    X, y = oversample.fit_resample(features_df, labels_df)
    del features_df
    del labels_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED
    )

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
        batch_size=BATCH_SIZE,
        validation_data=val_data,
        callbacks=callbacks,
    )
    if return_model:
        return history.history[monitor][-1], model, history
    else:
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

    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    SEED = params["seed"]
    MODEL_NAME = params["model_name"]
    TEST_SIZE = params["test_size"]
    EPOCHS = params["epochs"]
    BATCH_SIZE = params["batch_size"]
    EVAL_BATCH_SIZE = params["eval"]["batch_size"]

    if params["mode"] == "hyp_tuning":
        core_params = params["hyp_tuning"]
    else:
        core_params = params["train"]

    LEARNING_RATE = core_params["learning_rate"]
    ACTIVATION = core_params["activation"]
    OPTIMIZER = core_params['optimizer']

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )

    study.optimize(objective, n_trials=5, timeout=800)

    show_result(study)

    fig1 = plot_parallel_coordinate(study)
    fig1.write_html('parallel.html')
    fig2 = plot_param_importances(study)
    fig2.write_html('importance.html')

    # Make a table of GPU info
    g = nvgpu.gpu_info()
    df = pd.DataFrame.from_dict(g[0], orient="index", columns=["Value"])
    with open("gpu_info.txt", "w") as outfile:
        outfile.write(df.to_markdown())

    # run_model_training()
