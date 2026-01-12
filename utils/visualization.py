import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# -----------------------------
# Metrics (Notebook 03)
# -----------------------------
def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Returns a dict with Accuracy, Precision, Recall, F1-score
    """
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
    }


def metrics_table(metrics_list):
    """
    Takes a list of dicts from compute_metrics() and returns a DataFrame
    for the comparison table 
    """
    return pd.DataFrame(metrics_list)


# -----------------------------
# 2) Confusion Matrix (Notebook 04)
# -----------------------------
def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots confusion matrix 
    Returns (TN, FP, FN, TP) for interpretation text.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


# -----------------------------
# ROC + AUC (Notebook 04)
# -----------------------------
def roc_auc_values(y_true, y_prob):
    """
    Computes ROC curve and AUC :
    y_prob should be predict_proba(X)[:, 1]
    Returns fpr, tpr, auc_value
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = auc(fpr, tpr)
    return fpr, tpr, auc_value


def plot_roc_comparison(fpr1, tpr1, auc1, label1,
                        fpr2, tpr2, auc2, label2,
                        title="ROC Curve Comparison"):
    """
    Plot ROC comparison of two models (Logistic vs Random Forest)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f"{label1} (AUC = {auc1:.3f})")
    plt.plot(fpr2, tpr2, label=f"{label2} (AUC = {auc2:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()
