from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def tune_search(xtrain: np.ndarray, ytrain: np.ndarray, xtest:np.ndarray,
                groups, pipe: BaseEstimator, params: dict):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=sgkf,
        verbose=1,
        scoring='f1_macro'
    )

    grid.fit(xtrain, ytrain, groups=groups)
    ypred = grid.predict(xtest)
    yprob = grid.predict_proba(xtest)
    print(f"Model Best Parameters: {grid.best_params_}")
    print(f"Cross Validation F1-Macro Score: {grid.best_score_}")
    return grid.best_estimator_, ypred, yprob

def show_metrics(ytest: np.ndarray, ypred: np.ndarray, yproba: np.ndarray):
    _accuracy = accuracy_score(ytest, ypred)
    roc_score = roc_auc_score(ytest, yproba, multi_class='ovr')
    print(f"Accuracy and ROC AUC Score")
    print(f"=======================")
    print(f"Accuracy Score: {(_accuracy * 100):.3f}%")
    print(f"ROC AUC Score: {(roc_score * 100):.3f}%")
    print(f"=======================")
    print(classification_report(ytest, ypred))

def visualize_metrics(model, cm, xtest, ytest, name: str, labels: list):
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(1, 2, figsize=(13, 7)) 
    plt.suptitle(f'Metrics Visualizations: {name}')

    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax[0], xticklabels=['Legitimate', 'Smishing', 'Spam'],
                yticklabels=['Legitimate', 'Smishing', 'Spam'], linewidths=0.5, linecolor='black')
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('Actual Labels')

    yprob = model.predict_proba(xtest)
    ytest_bin = label_binarize(ytest, classes=[0, 1, 2])
    
    for i, cls in enumerate(labels):
        RocCurveDisplay.from_predictions(
            ytest_bin[:, i], yprob[:, i],
            name=f'{cls} vs Rest', 
            ax=ax[1]
        )
        
    ax[1].grid(True, alpha=0.8)
    ax[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax[1].legend(loc="lower right")
    ax[1].set_title('ROC Curve')
    
    plt.tight_layout()
    plt.show()


    