import pandas as pd
import numpy as np
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
from evaluation import evaluate_model

def logistic_regression(data_path="../train_data.csv"):
    # 读取数据
    df = pd.read_csv(data_path)

    # 特征与标签
    X = df.drop(columns=["ID", "encoded_label"])
    y = df["encoded_label"]


    # 参数搜索空间
    param_dist = {
        'C': loguniform(1e-4, 1e4),
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'penalty': ['l2'],
        'max_iter': [500, 1000, 2000]
    }

    # 定义模型和搜索器
    base_clf = LogisticRegression()
    random_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # 模型训练（直接用所有数据）
    random_search.fit(X, y)

    # 输出最优模型和参数
    print("Best Parameters from Random Search:")
    print(random_search.best_params_)

    # 获取训练好的最佳模型
    best_model = random_search.best_estimator_

    # 调用测试函数
    evaluate_model(best_model,classes=best_model.classes_,name='Logistic',result_dir='logistic_result')

    return best_model

best_model = logistic_regression()