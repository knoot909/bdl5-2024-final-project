import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix, 
    classification_report, roc_curve, precision_recall_curve, auc
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time

def evaluate_model_on_test(model, dataset):
    
    X_val = dataset.drop(columns=['target'])
    y_val = dataset['target']
    
    start_val = time.time()
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    end_val = time.time()
    val_time = end_val - start_val
    
    threshold = 0.62
    y_val_pred = (y_val_pred_prob >= threshold).astype(int)

    auc_val = roc_auc_score(y_val, y_val_pred_prob)
    recall = recall_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)
    fp_tp_ratio = cm[0, 1] / cm[1, 1] if cm[1, 1] != 0 else float('inf')
    clf_report = classification_report(y_val, y_val_pred)
    
    print(f"Час роботи моделі: {val_time}")
    print(f"AUC: {auc_val:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"FP/TP Ratio: {fp_tp_ratio:.4f}")
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix:")
    print(cm)

    sns.histplot(y_val_pred_prob[y_val == 0], color='red', label='Class 0 (val)', kde=True, stat='density', linewidth=0)
    sns.histplot(y_val_pred_prob[y_val == 1], color='green', label='Class 1 (val)', kde=True, stat='density', linewidth=0)
    plt.title(f'Probability Distribution')
    plt.legend()
    plt.show()
        
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend()
    plt.show()
        
    precision, recall, _ = precision_recall_curve(y_val, y_val_pred_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.legend()
    plt.show()
