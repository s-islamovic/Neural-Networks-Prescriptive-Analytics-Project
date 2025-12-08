from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = (preds > 0.5).astype(int)

    print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, preds))
    print("\nREPORT:\n", classification_report(y_test, preds))
    print("AUC:", roc_auc_score(y_test, preds))
