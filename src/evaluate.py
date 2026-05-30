from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred, model_name: str = "model"):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n===== {model_name} =====")
    print(metrics)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics