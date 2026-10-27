## Project Report

### Dataset
- The initial dataset used for the dry-run of initial analysis and modeling is a Philippine-based SMS Text Messages Dataset from Kaggle, with an initial 1,021 rows and 6 columns.
- THe data was subset into removing spam messages as the model might hallucinate on classifying both spam and scam. Thus, only focusing on the smishing messages rather than including spam messages.
- The initial dataset from Kaggle include labels such as `Legitimate`, `Scam`, and `Spam`
- This project requested and will use atleast 10,000 records of SMS Text Messagees Dataset from NTC.

### Data Wrangling and Cleaning
This section provides the in-depth process of the data preprocessing and cleaning procedure for the initial data. A pipeline was created for the cleaning, preprocessing, and feature extraction for the sms text messages. Moreover, through signal auditing on our current dataset. Therefore, the following values are audited as **Removed**, **Kept**, and **Extracted**

### Feature Auditing
- `masked_celphone_number` (Removed): The prefix of the contact number used to determine the sim carrier of the number.
- `hashed_cellphone_number` (Removed): Hashed value of the contact number, no analysis significance
- `date` (Removed): Used when extracting temporal features such as `month`, `year`, and `day` which is necessary for Exploratory Data Analysis 
- `text` (Kept): It is the Independent Variable (X) of the main fetaure of the analysis and for the smishing classification modeling.
- `carrier` (Extracted): It is the SIM Carrier derived from the `masked_cellphone_number`, used for Exploratory Data Analysis
- `target` (Kept): It is the Dependent Variable (Y) of the main feature of the analysis and the target for the classifier.
- `has_url` (Extracted): A signal feature extracted from `text` column and used if a specific text has an instance of a url link
- `word_count` (Extracted): A signal feature extracted from `text` through word tokenization process.
- `text_count` (Extracted): A signal feature extracted from `text` through overall text tokenization process.
- `has_money_mention` (Extracted): A signal feature extracted as boolean value from `text` and tells whether a specific text message has an instance of money.
- `has_phone_in_text` (Extracted): A signal feature extracted as boolean value from `text` and tells whether a specific text message has an instance of another phone number.
- `exclamation_count` (Extracted): A signal feature extracted through tokenization of exclamation marks, and acts as a frequency count of exclamations in an sms text messages.

### Text Preprocessing Audit
- Each sms text message in the `text` column was transformed through a preprocessing pipeline.
- Each Tagalog and English stopwords and leetspeaks are standardized and removed.
- Through the process of preprocessing, top level domains are removed, urls, and instances of money and masked through entity masking (e.g P1000 was converted to `[moneytoken]` or www.site.com to `[urltoken]`)

### Initial Modeling Audit
- This project utilizes two kinds on Naive Bayes model which is the MultinomialNB and the ComplementNB, and a more complex model using Random Forest Classifier
- Both Naive Bayes Model demonstrates a reasonable accuracy score and roc-auc score. However, when it comes to handling trade-offs between different metrics, the Complement Naive Bayes model demomnstrates a more balanced trade-offs between precision-recall, and f1 scoring.
- Complement Naive Bayes tend to reduced majority-class bias across all labels, on the other hand, Multinomial Naive Bayes is more prone to bias a more majority-class label.
- On the other hand, the Random Forest Classifier also demonstrated a decent accuracy score and roc-auc score, meaning Random Forests are accurate on classifying labels across different thresholds. However, its metrics also suffered some minor unbalanced tradeoffs between precision-recall, and f1 score.
- At the initial model creation, each model uses a fixed parameter, laplacian smoothing for NB Models to prevent zero probabilities within test data. 
- For Random Forests, it uses the default 100 decision trees to create an ensemble, and a log loss criterion for splitting.

### Hyperparameter Tuning Audit
- Both Naive Bayes Models used an alpha, or laplacian smoothing that prevents zero probabilities.
- CNB: 70.803% Accuracy with a bad trade-off metrics between precision-recall and f1 score.
- MNB: 70.073% Accuracy with a bad trade-off metrics between precision-recall and f1 score.
- The Random Forest Classifier used the following tuning parameters:
- 'rf__class_weight': None, 'rf__criterion': 'entropy', 
- 'rf__max_depth': 100, 'rf__max_features': 'log2', '
- rf__min_samples_leaf': 1, 'rf__min_samples_split': 4, 'rf__n_estimators': 200
- Random Forest Classifier: Achieved 82.432% Accuracy with a fair trade-off across classification metrics such as precision, recall, and f1 score, 