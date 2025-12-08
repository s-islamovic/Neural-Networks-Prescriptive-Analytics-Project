import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="/content/creditcard.csv"):
    data = pd.read_csv(path)

    X = data.drop("Class", axis=1)
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Count transactions per class
class_counts = data['Class'].value_counts()
total = class_counts.sum()
percentages = class_counts / total * 100

# Plot absolute counts with log scale
plt.figure(figsize=(6,4))
class_counts.plot(kind='bar', color=['skyblue','salmon'], log=True)
plt.xticks([0,1], ['Non-Fraud', 'Fraud'], rotation=0)
plt.ylabel('Number of Transactions (log scale)')
plt.title('Class Distribution (Log Scale for Visibility)')
plt.show()

# Plot percentages for easier interpretation
plt.figure(figsize=(6,4))
percentages.plot(kind='bar', color=['skyblue','salmon'])
plt.xticks([0,1], ['Non-Fraud', 'Fraud'], rotation=0)
plt.ylabel('Percentage of Transactions (%)')
plt.title('Class Distribution (Percentage)')
for i, p in enumerate(percentages):
    plt.text(i, p + 0.05, f"{p:.2f}%", ha='center')
plt.show()


