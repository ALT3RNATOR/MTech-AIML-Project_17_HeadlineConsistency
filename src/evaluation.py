"""Validation threshold selection and assignment metric calculation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def choose_threshold(y_true, scores) -> float:
    """Maximize validation Macro-F1; break ties with consistent-class F1."""
    y_true=np.asarray(y_true); scores=np.asarray(scores)
    candidates=np.unique(np.quantile(scores,np.linspace(0,1,1001)))
    best=(-1.0,-1.0,0.5)
    for threshold in candidates:
        pred=(scores>=threshold).astype(int)
        macro=f1_score(y_true,pred,average="macro",zero_division=0)
        consistent=f1_score(y_true,pred,pos_label=1,zero_division=0)
        candidate=(macro,consistent,float(threshold))
        if candidate>best: best=candidate
    return best[2]


def calculate_metrics(y_true, scores, threshold: float) -> dict[str,float]:
    """Compute required metrics with label 1 (`consistent`) as positive."""
    y_true=np.asarray(y_true); scores=np.asarray(scores); pred=(scores>=threshold).astype(int)
    return {
        "accuracy":float(accuracy_score(y_true,pred)),
        "precision":float(precision_score(y_true,pred,pos_label=1,zero_division=0)),
        "recall":float(recall_score(y_true,pred,pos_label=1,zero_division=0)),
        "macro_f1":float(f1_score(y_true,pred,average="macro",zero_division=0)),
        "roc_auc":float(roc_auc_score(y_true,scores)),
        "threshold":float(threshold),
    }
