# Hugging Face Space (Docker SDK) for the MedEvidence agent: the live chat page at "/".
# Built from the folder that deploy/hf-space/prepare.sh stages (code + the small abstract index).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf RERANK_DEVICE=cpu HOST=0.0.0.0 PORT=7860 HOME_PAGE=eval \
    LIMIT_STREAM=1 RATE_LIMIT_PER_MIN=6 RATE_LIMIT_BURST=3 \
    LLM_MODEL=Qwen/Qwen3-8B EMBEDDING_MODEL=bge-small-en-v1.5
WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the reranker and the embedding tokenizer into the image (no download at start-up).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('ncbi/MedCPT-Cross-Encoder'); \
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')" \
 && chmod -R a+rwX /opt/hf
ENV HF_HUB_OFFLINE=1

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
COPY --chown=user . /app
RUN mkdir -p .cache results/traces certs && chown -R user /app
USER user
EXPOSE 7860
CMD ["bash", "entrypoint.sh"]
