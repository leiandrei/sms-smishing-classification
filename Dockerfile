FROM python:3.14.3-slim

WORKDIR /sms-scam-classification

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/sms-scam-classification \
    MODEL_BUNDLE_PATH=/sms-scam-classification/model/bundled_pipeline.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY api/ ./api/
COPY src/ ./src/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]