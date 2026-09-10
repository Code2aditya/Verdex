FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Production inside the image: silences SQLAlchemy echo spam in Railway logs.
# (Overridable via Railway Variables; local dev keeps development default.)
ENV APP_ENV=production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models/checkpoints

# Bake EasyOCR + InsightFace weights into the image so Railway restarts
# don't re-download ~500MB on the first request (non-fatal if it fails).
RUN python warmup_models.py || echo "warmup incomplete - runtime download fallback remains"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]