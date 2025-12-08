#baseline model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_dim, lr=0.001, dropout_rate=0.2, hidden_units=32):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

#train baseline
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data
from baseline_model import build_model
from genetic_algorithm_optimization import GeneticAlgorithm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from evaluation import evaluate_model

def plot_roc_pr(y_test, y_pred_probs):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.subplot(1,2,2)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data(path="/content/creditcard.csv")  # adjust path if needed

    # Run GA to get best hyperparameters
    ga = GeneticAlgorithm(X_train, y_train)
    best_params = ga.evolve()

    print("\nTraining final model with GA-optimized parameters...")
    model = build_model(
        input_dim=X_train.shape[1],
        lr=best_params["lr"],
        dropout_rate=best_params["dropout_rate"],
        hidden_units=best_params["hidden_units"]
    )

    # Define class weights to handle imbalance
    class_weight = {0: 1, 1: 50}  # gives more weight to fraud class

    model.fit(
        X_train, y_train,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        validation_split=0.2,
        verbose=1,
        class_weight=class_weight
    )

    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
