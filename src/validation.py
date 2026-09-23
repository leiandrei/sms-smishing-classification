
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import BaseEstimator
import scipy.sparse as sp
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
    
TARGET_NAMES = ['Legitimate', 'Smishing', 'Spam']

def model_check(model: BaseEstimator, df: pd.DataFrame, 
                name: str, n: int | None = 10):
    """Validates the predicted values against the actual values from the test df and predicted"""
    sampled = df.sample(n, random_state=42) if n is not None else df

    texts = sampled['text_transformed'].values   
    predictions = model.predict(texts)            
    actuals = sampled['target'].values

    correct = 0
    print(f"Predicted vs Actuals: ({name})")
    print("================================")

    for text, pred, act in zip(texts, predictions, actuals):
        flag = "Matched" if pred == act else "Not Matched"
        if pred == act:
            correct += 1
        print(f"[{flag}] Predicted: {TARGET_NAMES[pred]:10s} | Actual: {TARGET_NAMES[act]:10s} | {text[:60]}")

    print(f"\nNumber of correct predictions: {correct}/{len(sampled)}")

def get_clf_and_names(model: BaseEstimator):
    if not hasattr(model, 'named_steps'):
        raise ValueError("Expected a fitted Pipeline containing a vectorizer step.")

    clf = model.named_steps.get('clf', list(model.named_steps.values())[-1])
    vec_step = next(
        (s for s in model.named_steps.values() if hasattr(s, 'get_feature_names_out')),
        None
    )
    if vec_step is None:
        raise ValueError("No vectorizer step found in this pipeline.")

    return clf, np.array(vec_step.get_feature_names_out())

def top_words_per_class_nb(model, class_names, top_n=15):
    """Extracts the key importance (words) from each class category from Naive Bayes Model"""
    clf, feature_names = get_clf_and_names(model)
    for idx, name in enumerate(class_names):
        others = [c for c in range(len(class_names)) if c != idx]
        log_odds = clf.feature_log_prob_[idx] - np.mean(clf.feature_log_prob_[others], axis=0)
        top_idx = np.argsort(log_odds)[::-1][:top_n]
        print(f"--- Top {top_n} words for {name} (NB log-odds) ---")
        for i in top_idx:
            print(f"{feature_names[i]:25s}  log-odds={log_odds[i]:.3f}")
        print()

def top_words_overall_rf(model, top_n=15):
    clf, feature_names = get_clf_and_names(model)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f"--- Top {top_n} words overall (RF feature importance) ---")
    for i in top_idx:
        print(f"{feature_names[i]:25s}  importance={importances[i]:.4f}")
