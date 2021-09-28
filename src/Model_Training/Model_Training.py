#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import os
import yaml
import tf2onnx
import matplotlib.pyplot as plt

### Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)
### Taken from https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
### More in here : https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
### TODO Take a good look at the metric
def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



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

    # FIXME Find this from code
    N_FEATURES = len(features_df.columns)
    N_LABELS = 14

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
                  metrics=[custom_f1])

    # model.fit(train_data, callbacks=callbacks, epochs=1)
    history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)
    print("Model training done")

    # Make a plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_plot.png')

    # Convert model to onnx
    # spec = (tf.TensorSpec((None, N_FEATURES), tf.float32, name="input"),)
    output_path = MODEL_NAME + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=output_path)
    print("Converted and saved the model to ONNX file")

if __name__ == "__main__":
    run_model_training()
