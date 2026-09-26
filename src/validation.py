from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import BaseEstimator
import scipy.sparse as sp
import pandas as pd
import numpy as np
    
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

def top_words_per_class_nb(model: BaseEstimator, class_names: list[str], top_n: int = 15):
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

def top_words_overall_rf(model: BaseEstimator, top_n: int = 15):
    clf, feature_names = get_clf_and_names(model)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f"--- Top {top_n} words overall (RF feature importance) ---")
    for i in top_idx:
        print(f"{feature_names[i]:25s}  importance={importances[i]:.4f}")

def top_words_per_class_rf(model: BaseEstimator, xtest, ytest, class_names: list[str], top_n: int = 15):
    """Extracts the key importance (words) for each class category from a Random Forest Model"""
    clf, feature_names = get_clf_and_names(model)

    # 1. Transform the raw text into the feature matrix first
    # model[:-1] slices the pipeline to get everything except the final classifier
    preprocessor = model[:-1]
    X_test_transformed = preprocessor.transform(xtest)
    
    # Scikit-learn's permutation_importance requires sparse matrices to be in CSC format
    if sp.issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()

    print("Top words for each class in Random Forests:")
    print("==============================================")
    
    # 2. Iterate through each class to calculate one-vs-rest importance
    for class_idx, name in enumerate(class_names):
        
        # Create a custom scorer that treats the current class as 1 and others as 0
        def class_f1(y_true, y_pred, c_idx=class_idx):
            return f1_score(y_true == c_idx, y_pred == c_idx, zero_division=0)
            
        scorer = make_scorer(class_f1)
        
        # 3. Run permutation on the isolated classifier using the transformed word features
        # Note: n_repeats is lowered to 5 to save compute time on high-dimensional text data
        results = permutation_importance(
            clf, X_test_transformed, ytest, 
            n_repeats=5, random_state=42, scoring=scorer
        )
        
        print(f"\n--- Top {top_n} words for {name} ---")
        
        # 4. Correctly reference the results object and feature_names array
        top_indices = results.importances_mean.argsort()[::-1][:top_n]
        
        for idx in top_indices:
            print(f"{feature_names[idx]:25s}: {results.importances_mean[idx]:.4f} +/- {results.importances_std[idx]:.4f}")

        




