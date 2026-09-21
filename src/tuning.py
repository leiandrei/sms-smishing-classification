from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def tune_search(xtrain: np.ndarray, ytrain: np.ndarray, pipe: BaseEstimator, params: dict):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        cv=sgkf,
        verbose=1,
        n_jobs=-1,
        scoring='macro_f1'
    )

    grid.fit(xtrain, ytrain)
    print(f"Model Best Parameters: {grid.best_params_}")
    print("Best CV macro F1:", grid.best_score_)
    return grid.best_estimator_

def plot_matrix(ytest: np.ndarray, ypred: np.ndarray):

    sns.set_style('whitegrid')
    cm = confusion_matrix(ytest, ypred)    
    ticklabels = ['Legitimate', 'Smishing', 'Spam']
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(cm, annot=True, cmap='magma', fmt='d', 
                xticklabels=ticklabels, yticklabels=ticklabels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

def plot_curve(model: BaseEstimator, xtest: np.ndarray, ytest: np.ndarray, name: str):

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = ['Legitimate', 'Smishing', 'Spam']

    yprob = model.predict_proba(xtest)
    ytest_bin = label_binarize(ytest, classes=[0, 1, 2])

    for i, cls in enumerate(labels):
        RocCurveDisplay.from_predictions(
            ytest_bin[:, i], yprob[:, i],
            name=f'{cls} vs Rest', ax=ax 
        )
        
    ax.grid(True, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.legend()
    ax.set_title(f'ROC Curve: {name}')
    plt.tight_layout()


def show_metrics(ytest: np.ndarray, ypred: np.ndarray, ypred_proba):

    roc_score = roc_auc_score(ytest, ypred_proba, multi_class='ovr', average='macro')
    accuracy = accuracy_score(ytest, ypred)

    print('=== Classifier Metrics ===')
    print(f"ROC AUC Score: {roc_score:.5f}")
    print(f"Accuracy Score: {accuracy:.5f}")
    print('==========================\n')
    print("Classification Report:")

    print(classification_report(ytest, ypred))
    print("========================================")


    