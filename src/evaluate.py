import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

def evaluate_model():
    model = tf.keras.models.load_model("models/baseline_model.h5")

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    preds = model.predict(X_test).ravel()

    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Baseline Model")
    plt.legend()
    plt.savefig("results/baseline_roc.png")

    cm = confusion_matrix(y_test, preds > 0.5)
    print("Confusion Matrix:\n", cm)

    np.savetxt("results/predictions.csv", preds, delimiter=",")

if __name__ == "__main__":
    evaluate_model()
