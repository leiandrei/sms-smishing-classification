## Text or Trap? - SMS Phishing Detection in Filipino SMS Messages using Naive Bayes and Ensemble-based Random Forest Classifiers.
A partial fullfilment for the final requirement for the elective subject COMP 034 - Introduction to Data Science

## Proponents
- Callanga, Aaron Dave D.
- Domaoal, Lei Andrei E.
- Monton, Sean Paul M.

## Overview
This project aims develop and train classifier models to detect Phishing schemes in SMS Text Messages in a localized-Filipino (Tagalog and Taglish context). Ultimately, this project
utilizes varied Naive Bayes models such as: Complement and Multinomial Naive Bayes, as a locally-interpretable baseline model. A Random Forest Classifier for complex or black-box model. The model was deployed
under a FastAPI service, and a web application for demonstration purposes.

## Dataset Source
The dataset was gathered across different sources

| Source | Description | Records | Labels |
| :--- | :--- | ---: | :--- |
| Philippine Spam/Scam SMS (BwandoWando, 2026) | Spam and scam messages personally received by the author, with some legitimate messages | 1,021 | Legitimate, Scam, Spam |
| Tagalog-SMS (ZERO, 2025) | Personal collection of all text messages received by the author over a period | 3,454 | spam, ads, gov, notifs, otp |
| Text or Trap | Screenshots and copies of text messages sent by contributors | 61 | Legit, Scam |
| NTC request | SMS records requested from the National Telecommunications Commission | 500 | Scam |

*Table I. Sources of Data*

## Methodology

The end-to-end classification pipeline follows six structured stages:

| Process | Technique | Configuration & Purpose |
| :--- | :--- | :--- |
| **Text Preprocessing & Feature Engineering** | Regex Normalization, Stopword Removal & Heuristic Extraction | Normalizes Unicode/leetspeak, masks URLs, phone numbers, and currency into tokens, removes English/Tagalog stopwords, and extracts 8 binary flags and 5 numerical text statistics |
| **Splitting Criteria** | Stratified Group K-Fold (`StratifiedGroupKFold`) | `n_splits=5`, `shuffle=True`, `random_state=42`, grouped by `dedup_group_id` to preserve class distribution while preventing duplicate SMS leakage across folds |
| **Text Vectorization** | TF-IDF Vectorizer (`TfidfVectorizer`) | `ngram_range=(1, 4)`, `min_df=2`, `max_df=0.5`, `sublinear_tf=True` to capture 1-to-4-gram phishing phrases while filtering rare noise and overly frequent terms |
| **Feature Scaling** | Min-Max Scaling (`MinMaxScaler`) | `clip=True` to bound numerical features within `[0, 1]` for compatibility with Naive Bayes estimators |
| **Model Training** | Multinomial NB, Complement NB & Random Forest | `MultinomialNB(alpha=0.1)`, `ComplementNB(alpha=0.1, norm=False)`, and `RandomForestClassifier(n_estimators=200, max_depth=200, criterion='entropy')` |
| **Statistical Significance** | McNemar's Test | Pairwise non-parametric hypothesis test on prediction contingency tables to determine whether performance differences between classifiers are statistically significant |

*Table II. Experimental Methodology and Pipeline Configuration*

## Initial Model Creation (Untuned Models)

Baseline performance evaluated on the holdout test fold ($n = 469$: 278 Legitimate, 191 Smishing):

| Model | Accuracy | ROC AUC | Smishing Precision | Smishing Recall | Smishing F1-Score | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MNB`)** | 95.74% | **97.89%** | 0.94 | 0.95 | 0.95 | 0.96 |
| **Complement Naive Bayes (`CNB`)** | **96.38%** | **97.89%** | 0.94 | **0.97** | **0.96** | **0.96** |
| **Random Forest (`RF`)** | 95.31% | 97.47% | **0.95** | 0.94 | 0.94 | 0.95 |

*Table III. Overall Performance Summary of Untuned Baseline Models*

### Per-Class Classification Breakdown

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MNB`)** | Legitimate (`0`) | 0.97 | 0.96 | 0.96 | 278 |
| | Smishing (`1`) | 0.94 | 0.95 | 0.95 | 191 |
| **Complement Naive Bayes (`CNB`)** | Legitimate (`0`) | 0.98 | 0.96 | 0.97 | 278 |
| | Smishing (`1`) | 0.94 | 0.97 | 0.96 | 191 |
| **Random Forest (`RF`)** | Legitimate (`0`) | 0.96 | 0.96 | 0.96 | 278 |
| | Smishing (`1`) | 0.95 | 0.94 | 0.94 | 191 |

*Table IV. Per-Class Precision, Recall, and F1-Score for Untuned Models*

## Tuned Models

Post-tuning performance evaluated on the same holdout test fold ($n = 469$: 278 Legitimate, 191 Smishing):

| Model | Accuracy | ROC AUC | Smishing Precision | Smishing Recall | Smishing F1-Score | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MNB`)** | **96.59%** | **98.51%** | **0.95** | 0.96 | **0.96** | **0.96** |
| **Complement Naive Bayes (`CNB`)** | 96.38% | 98.42% | 0.94 | **0.97** | **0.96** | **0.96** |
| **Random Forest (`RF`)** | 96.16% | 98.02% | **0.95** | 0.95 | 0.95 | **0.96** |

*Table V. Overall Performance Summary of Tuned Models*

### Per-Class Classification Breakdown

| Model | Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MNB`)** | Legitimate (`0`) | 0.97 | 0.97 | 0.97 | 278 |
| | Smishing (`1`) | 0.95 | 0.96 | 0.96 | 191 |
| **Complement Naive Bayes (`CNB`)** | Legitimate (`0`) | 0.98 | 0.96 | 0.97 | 278 |
| | Smishing (`1`) | 0.94 | 0.97 | 0.96 | 191 |
| **Random Forest (`RF`)** | Legitimate (`0`) | 0.97 | 0.97 | 0.97 | 278 |
| | Smishing (`1`) | 0.95 | 0.95 | 0.95 | 191 |

*Table VI. Per-Class Precision, Recall, and F1-Score for Tuned Models*

### Untuned vs. Tuned Performance Comparison

| Model | Untuned Accuracy | Tuned Accuracy | Accuracy Gain | Untuned ROC AUC | Tuned ROC AUC | ROC AUC Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (`MNB`)** | 95.74% | **96.59%** | **+0.85%** | 97.89% | **98.51%** | **+0.62%** |
| **Complement Naive Bayes (`CNB`)** | **96.38%** | 96.38% | 0.00% | 97.89% | 98.42% | +0.53% |
| **Random Forest (`RF`)** | 95.31% | 96.16% | **+0.85%** | 97.47% | 98.02% | +0.55% |

*Table VII. Impact of Hyperparameter Tuning Across Models*

## Statistical Significance (McNemar's Test)

Pairwise exact McNemar's tests were conducted at a significance threshold of $\alpha = 0.05$ to evaluate whether prediction differences between classifiers are statistically meaningful:

| Pairwise Comparison (Model A vs. Model B) | Model A Accuracy | Model B Accuracy | $p$-value | Significance ($\alpha = 0.05$) | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`CNB` vs. `MNB`** | 96.59% | 96.59% | 1.0000 | Not Significant | No significant difference in performance |
| **`CNB` vs. `RF`** | 96.59% | 95.74% | 0.2891 | Not Significant | No significant difference in performance |
| **`MNB` vs. `RF`** | 96.59% | 95.74% | 0.2188 | Not Significant | No significant difference in performance |

*Table VIII. Pairwise McNemar's Statistical Significance Test Results*


