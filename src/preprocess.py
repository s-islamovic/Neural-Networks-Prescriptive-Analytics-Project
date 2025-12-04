import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create processed data folder
    os.makedirs("data/processed", exist_ok=True)

    # Save NumPy arrays
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/X_test.npy", X_test)

    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing complete. Files saved to data/processed/")

if __name__ == "__main__":
    load_and_preprocess()
