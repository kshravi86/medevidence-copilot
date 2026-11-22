# MedEvidence Copilot: API (agent) and Streamlit UI share one image; docker-compose picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf RERANK_DEVICE=cpu HOST=0.0.0.0 PORT=8100
WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the reranker and the embedding tokenizer into the image (no download at start-up).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('ncbi/MedCPT-Cross-Encoder'); \
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')"

COPY src ./src
COPY app/server.py app/pipeline.py app/streamlit_app.py ./app/
COPY app/static ./app/static
COPY scripts ./scripts
COPY eval ./eval

# Not in the image (mounted at run time): .env (keys), certs/ (endpoint CAs), data/ (corpus + index),
# .cache/ (LLM response cache + sessions), results/traces (OpenTelemetry spans).
RUN useradd -m -u 10001 app && mkdir -p .cache results/traces && chown -R app /app
USER app
EXPOSE 8100 8501
CMD ["python", "app/server.py"]
