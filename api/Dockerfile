FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential coinor-cbc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# The `extras` group is not optional in practice: soccerdata drives the fbref connector
# (tackles/interceptions/blocks -> DefCon), youtube-transcript-api drives transcripts (and
# every yt_* feature behind them), sentence-transformers backs EMBEDDING_PROVIDER=local,
# and sqlite-vec backs vector search. Without it those sources yield nothing, quietly.
# CPU-only torch first, or pip drags in ~2.5GB of CUDA wheels this container cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[extras]"

EXPOSE 8000
CMD ["uvicorn", "fplai.main:app", "--host", "0.0.0.0", "--port", "8000"]
