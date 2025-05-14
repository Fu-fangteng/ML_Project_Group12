import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

def evaluate_model(model,classes, name,result_dir="evaluation_results"):
    """
    Evaluate the model with metrics like accuracy, precision, recall, f1, ROC curve, and confusion matrix.
    Supports any classifier that has `predict` and `predict_proba` methods.

    Parameters:
    - model: Trained model to evaluate (e.g., LogisticRegression, SVC, XGBClassifier, etc.).
    - X_test: Test features.
    - y_test: True labels for the test data.
    - classes: List of class labels (or model.classes_ if available).
    - result_dir: Directory to save the evaluation results (default is 'evaluation_results').
    """
    # Create result directory if not exists
    os.makedirs(result_dir, exist_ok=True)

    df_test = pd.read_csv("../test_data.csv")  # 测试集路径
    X_test = df_test.drop(columns=["ID", "encoded_label"])
    y_test = df_test["encoded_label"]
    n_classes = len(classes)

    # Binarize labels for ROC curve if applicable
    y_bin = label_binarize(y_test, classes=classes)

    # Predict the labels and probabilities
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)


    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save metrics to file
    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("模型在测试集上的评估结果：\n")
        f.write(f"Accuracy：{acc:.4f}\n")
        f.write(f"Precision：{prec:.4f}\n")
        f.write(f"Recall：{rec:.4f}\n")
        f.write(f"F1 score：{f1:.4f}\n\n")


    fpr, tpr, roc_auc = dict(), dict(), dict()
    with open(os.path.join(result_dir, "metrics.txt"), "a", encoding="utf-8") as f:
        f.write("各类 AUC 值：\n")
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            f.write(f"  Class {classes[i]}: AUC = {roc_auc[i]:.2f}\n")

        # Save ROC curve plot
        plt.figure(figsize=(6, 5))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"{name} Test Set - Multi-class ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "roc_curve.png"))
        plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f'{name} Confusion Matrix - Per Class')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(result_dir, "confusion_matrix_per_class.png"))
    plt.close()

