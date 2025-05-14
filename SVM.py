import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
if isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


svm_model = SVC(
    kernel='rbf',          
    C=1.0,                
    gamma='scale',    
    decision_function_shape='ovr',  
    random_state=42,
    probability=True      
)

cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)

print("\nEvaluation:")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1_score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nreport:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('confusion_matrix')
plt.xlabel('predict_label')
plt.ylabel('true_label')
plt.show()


svm = OneVsRestClassifier(SVC(
    kernel='rbf', 
    C=1.0, 
    gamma='scale',
    probability=True,  
    random_state=42
))
svm.fit(X_train, y_train)
y_score = svm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve (Binary Classification)')
plt.legend(loc="lower right")
plt.show()