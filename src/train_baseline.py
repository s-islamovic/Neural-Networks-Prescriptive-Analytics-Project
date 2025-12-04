import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model():
    model = models.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train():
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")

    model = build_model()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=2048,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True
        )]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/baseline_model.h5")

    print("Baseline model trained and saved.")

if __name__ == "__main__":
    train()
