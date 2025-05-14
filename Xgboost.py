import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, accuracy_score, f1_score, auc
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
if isinstance(y.iloc[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3,           
    random_state=42,           
    stratify=y                 
)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'multi:softmax',
    'num_class': len(set(y)),
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'seed': 42
}

evals = [(dtrain, 'train')]
num_round = 1000  

bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=evals,
    early_stopping_rounds=20,  
    verbose_eval=10            
)

y_pred = bst.predict(dtest)
print("\nEvaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1_score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print("\nreport:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('confusion_matrix')
plt.xlabel('prediction_label')
plt.ylabel('true_label')
plt.show()

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train")],
    early_stopping_rounds=10
)
y_pred = model.predict(dtest)
y_pred_class = [1 if x > 0.5 else 0 for x in y_pred]  
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

