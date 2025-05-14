import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# 创建保存目录
os.makedirs("results/random_forest", exist_ok=True)

# 读取数据
df = pd.read_csv("preprocessing/processed_data_label_encoding.csv")

# 特征与标签
X = df.drop(columns=["ID", "encoded_label"])
y = df["encoded_label"]
classes = np.unique(y)
n_classes = len(classes)

# 标签二值化用于 ROC
y_bin = label_binarize(y, classes=classes)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# 使用随机森林进行多分类
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 数据集集合
datasets = {
    "Train": (X_train, y_train, y_train_bin),
    "Test": (X_test, y_test, y_test_bin),
    "All": (X, y, y_bin)
}

# 评估 + 可视化函数
for name in tqdm(datasets, desc="Evaluating datasets"):
    X_set, y_true, y_bin_true = datasets[name]
    y_pred = clf.predict(X_set)  # 直接预测类别标签
    y_score = clf.predict_proba(X_set)

    # 评估指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n[{name} Set]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{name} Set - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/random_forest/{name.lower()}_confusion_matrix.png")
    plt.close()

    # ROC Curve for multiclass
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每个类别的 ROC 曲线和 AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 输出每个类别的 AUC 分数
    print(f"AUC for {name} set:")
    for i in range(n_classes):
        print(f"  Class {classes[i]}: AUC = {roc_auc[i]:.2f}")

    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} Set - Multi-class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/random_forest/{name.lower()}_roc_curve.png")
    plt.close()
