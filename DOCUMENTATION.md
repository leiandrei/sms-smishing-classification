# Technical Documentation — Text or Trap

## 1. System Overview

This repository implements an end-to-end machine learning pipeline and containerized REST API for detecting smishing (SMS phishing) messages written in English, Tagalog, and Taglish within the Philippine telecommunications context.

The system consists of four main layers:
1. **Preprocessing & Feature Engineering (`src/preprocessing.py`):** Normalizes raw SMS messages, masks sensitive entities (URLs, phone numbers, currency), removes English and Tagalog stopwords, maps Philippine telco prefixes, and extracts 13 domain-specific heuristic signals.
2. **Model Serialization (`model/bundled_pipeline.joblib`):** Packages a single fitted `ColumnTransformer` alongside three trained classifiers (`MultinomialNB`, `ComplementNB`, and `RandomForestClassifier`) into a compressed `.joblib` artifact.
3. **Inference API (`api/main.py`):** Serves independent prediction and hyperparameter inspection endpoints via FastAPI and Uvicorn.
4. **Web Interface (`app/`):**

---

## 2. Environment & Dependencies

* **Python Version:** `Python 3.14.3` (Docker base image: `python:3.14.3-slim`)
* **Environment & Package Manager:** `uv` (`.venv` virtual environment)
* **Container Runtime:** Docker & Docker Compose (`nginx:alpine` for static frontend serving)

### Core Libraries

| Category | Library | Role in Project |
| :--- | :--- | :--- |
| **Machine Learning** | `scikit-learn` | `TfidfVectorizer`, `MinMaxScaler`, `ColumnTransformer`, `StratifiedGroupKFold`, `MultinomialNB`, `ComplementNB`, `RandomForestClassifier` |
| **Data Processing** | `pandas`, `numpy`, `scipy` | Tabular data wrangling, sparse matrix operations, and single-row inference DataFrame construction |
| **NLP** | `nltk` | English stopword corpus (`nltk.corpus.stopwords`) combined with a custom Tagalog stopword lexicon (with automatic fallback to `sklearn` English stopwords) |
| **Serialization** | `joblib` | Compressed (`compress=3`) persistence and deserialization of the shared preprocessor and fitted models |
| **API & Validation** | `fastapi`, `uvicorn`, `pydantic` | REST routing, ASGI server execution, CORS middleware, and request/response schema validation |
| **Testing** | `pytest`, `httpx` | Automated endpoint, schema, boundary, and error-handling tests via FastAPI's `TestClient` |

---

## 3. Project Structure

```text
sms-scam-classification/
├── api/
│   ├── main.py                              # FastAPI application and prediction/parameter endpoints
│   └── testing.py                           # Pytest integration and contract test suite
├── app/                                     # Web demonstration UI
├── docs/
│   └── Initial Report - EDA.pdf             # Exploratory Data Analysis report
├── model/
│   └── bundled_pipeline.joblib              # Serialized ColumnTransformer + MNB, CNB, and RF models
├── notebooks/
│   ├── review/
│   │   ├── 01_review_queue.ipynb            # Manual data labeling & curation queue
│   │   ├── 02_review_batch.ipynb            # Batch verification notebook
│   │   └── tasks.md                         # Curation task notes
│   ├── 01_data_wrangling.ipynb              # Dataset merging, deduplication, and cleaning
│   ├── 02_exploratory_da.ipynb              # Statistical & visual exploratory data analysis
│   ├── 03_model_creation.ipynb              # Untuned baseline model training
│   ├── 04_model_tuning.ipynb                # Hyperparameter optimization
│   ├── 05_statistical_significance.ipynb    # Pairwise McNemar's contingency tests
│   ├── 06_model_evaluation.ipynb            # Holdout evaluation and model bundling
│   ├── rf_6_trees.png                       # Random Forest sub-tree visualization
│   └── rf_single_tree.png                   # Single decision tree visualization
├── src/
│   ├── get_data.py                          # Data ingestion utilities
│   ├── preprocessing.py                     # Text normalization, telco mapping, and feature extraction
│   ├── tuning.py                            # Hyperparameter search utilities
│   └── validation.py                        # Group-aware cross-validation utilities
├── .dockerignore                            # Excludes .venv, data/, and notebooks from Docker builds
├── .gitignore                               # Excludes .venv, .env, data/, *.csv, caches, and local task logs
├── docker-compose.yml                       # Multi-container orchestration (FastAPI + Nginx UI)
├── Dockerfile                               # Container build instructions for the FastAPI service
├── DOCUMENTATION.md                         # Technical system and API documentation
├── LICENSE                                  # Project license
├── README.md                                # Project summary, methodology, and benchmark results
└── requirements.txt                         # Pinned Python package dependencies
```

> **Note:** Local datasets (`data/`), all `.csv` files (`*.csv`), environment secrets (`.env`), virtual environments (`.venv`), and internal task trackers (`TASKS.md`, `REVISIONS.md`) are excluded from version control via `.gitignore`.

---

## 4. Preprocessing & Feature Engineering (`src/preprocessing.py`)

Both batch training (`transform()`) and single-message API inference (`extract_inference_features()`) apply the exact same deterministic pipeline:

1. **Text Normalization (`transform_text` & `remove_stopwords`):**
   * Applies Unicode NFKC normalization and lowercasing.
   * Replaces URLs, Philippine mobile numbers (`+639...` / `09...`), and monetary values (`PHP`, `₱`, `P`) with standardized tokens (`urltoken`, `phonetoken`, `moneytoken`).
   * Decodes common leetspeak substitutions (e.g., `gc4sh`, `bd0`, `sm4rt`) while preserving ordinals and timestamps (`1st`, `10am`).
   * Strips punctuation and filters out combined English and Tagalog stopwords.
2. **Philippine Telco Carrier Mapping (`update_carrier`):**
   * Maps 4-digit mobile prefixes (`09XX`) and sender IDs to `Smart`, `Globe`, `DITO`, `GOMO`, `Alphanumeric Sender`, or `Shortcode`.
3. **Model Input Feature Set (14 Input Columns):**
   * **Text Feature (1):** `text` — vectorized via `TfidfVectorizer(ngram_range=(1, 4), min_df=2, max_df=0.5, sublinear_tf=True)`.
   * **Binary Signal Flags (8, passthrough):** `has_url`, `has_money_mention`, `has_phone_in_text`, `has_otp`, `urgency`, `has_leetspeaks`, `typo_squatting`, `has_url_shortened`.
   * **Numerical Text Statistics (5, scaled via `MinMaxScaler(clip=True)`):** `exclamation_count`, `text_count`, `word_count`, `punctuation_count`, `uppercase_ratio`.

---

## 5. Model Bundle Architecture (`model/bundled_pipeline.joblib`)

To avoid redundant TF-IDF tokenization across three separate pipelines, `bundled_pipeline.joblib` stores **one shared fitted `ColumnTransformer`** alongside the **three fitted estimators**:

```python
{
    "preprocessor": ColumnTransformer(...),
    "models": {
        "mnb": MultinomialNB(alpha=0.1, fit_prior=True),
        "cnb": ComplementNB(alpha=0.1, norm=False),
        "rf": RandomForestClassifier(
            class_weight=None,
            criterion="entropy",
            max_depth=200,
            max_features="log2",
            min_samples_split=6,
            min_samples_leaf=1,
            n_estimators=200,
            random_state=42,
        ),
    },
    "target_names": ["Legitimate (ham)", "Smishing (scam)"],
    "feature_cols": ["text", ...],
    "metadata": {
        "sklearn_version": "<version>",
        "created_at": "<ISO-8601 UTC timestamp>",
    },
}
```

> **Note:** At API startup, `rf.n_jobs` is set to `1` to prevent per-request thread-pool overhead during single-row HTTP inference, while `/rf/parameters` reports the original training configuration.

---

## 6. Setup & Deployment

### Option A: Local Development Setup

1. **Create and activate the virtual environment:**
   ```bash
   uv venv --python 3.14.3
   # Git Bash (Windows):
   source .venv/Scripts/activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies and NLTK stopwords:**
   ```bash
   uv pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords')"
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   * Interactive Swagger UI: `http://127.0.0.1:8000/docs`
   * ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### Option B: Docker & Docker Compose (Full Stack)

Ensure Docker Desktop (Linux containers engine) is running, then build and launch both the API (`port 8000`) and Nginx frontend (`port 3000`):

```bash
docker compose up --build -d
```

* **Frontend Web App:** `http://localhost:3000`
* **FastAPI Service:** `http://localhost:8000`
* **Reload updated model bundle without rebuilding:**
  ```bash
  docker compose restart api
  ```
* **Stop services:**
  ```bash
  docker compose down
  ```

### Dataset Regeneration (For Model Retraining)

Because `data/` and `*.csv` files are excluded via `.gitignore`, inference works out-of-the-box using the pre-packaged `model/bundled_pipeline.joblib`. To retrain models from scratch, run `src/get_data.py` followed by `notebooks/01_data_wrangling.ipynb` to rebuild the local `data/` directory.

### Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_BUNDLE_PATH` | `model/bundled_pipeline.joblib` | Path to the serialized `.joblib` model bundle loaded at API startup. |

---

## 7. API Reference

### Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Returns API status and available endpoint paths. |
| `GET` | `/health` | Health check confirming the `.joblib` bundle loaded without `scikit-learn` version mismatches. |
| `GET` | `/mnb/parameters` | Returns hyperparameters for `MultinomialNB`. |
| `GET` | `/cnb/parameters` | Returns hyperparameters for `ComplementNB`. |
| `GET` | `/rf/parameters` | Returns hyperparameters for `RandomForestClassifier`. |
| `POST` | `/mnb/predict` | Classifies a raw SMS message using `MultinomialNB`. |
| `POST` | `/cnb/predict` | Classifies a raw SMS message using `ComplementNB`. |
| `POST` | `/rf/predict` | Classifies a raw SMS message using `RandomForestClassifier`. |

### Prediction Request Schema (`POST /{model}/predict`)

* **Content-Type:** `application/json`
* **Constraints:** `sms_message` must be a non-empty string between `1` and `500` characters.

```json
{
  "sms_message": "URGENT: Verify your GCash account at [http://kycupdategcash.online](http://kycupdategcash.online)"
}
```

### Prediction Response Schema (`200 OK`)

```json
{
  "model": "MultinomialNB",
  "label": "Smishing (scam)",
  "confidence": 0.872042,
  "probabilities": {
    "Legitimate (ham)": 0.127958,
    "Smishing (scam)": 0.872042
  }
}
```

### Example `curl` Requests

```bash
# Health check
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

# Predict with Complement Naive Bayes
curl -X POST "[http://127.0.0.1:8000/cnb/predict](http://127.0.0.1:8000/cnb/predict)" \
  -H "Content-Type: application/json" \
  -d '{"sms_message": "Your BDO account is suspended. Update your OTP via bit.ly/bdo-verify now!"}'

# Inspect Random Forest parameters
curl [http://127.0.0.1:8000/rf/parameters](http://127.0.0.1:8000/rf/parameters)
```

### HTTP Error Codes

| Status Code | Cause |
| :--- | :--- |
| `422 Unprocessable Entity` | Missing `sms_message` field, empty string (`""`), `null`, or string exceeding `500` characters. |
| `500 Internal Server Error` | Unexpected runtime exception during feature extraction or model inference. |
| `503 Service Unavailable` | Model bundle file missing, corrupted, or trained with an incompatible `scikit-learn` version. |

---

## 8. Automated Testing (`api/testing.py`)

Run the `pytest` suite from the repository root to verify endpoint routing, response schemas, feature extraction columns, input length boundaries, stopword-only edge cases, and `503` degradation handling:

```bash
python -m pytest api/testing.py -v
```