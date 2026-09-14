FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VPN_MODEL_PATH=/app/models/xgb_15s.joblib

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Build a deterministic demo artifact. A failed training run must fail the image build.
RUN python train.py --window 15 --output "${VPN_MODEL_PATH}"

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "gradio_app.py"]
