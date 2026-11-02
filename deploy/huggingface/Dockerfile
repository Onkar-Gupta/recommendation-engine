# Hugging Face Space image. Build context is the repo root (or the staged copy made by push.ps1).
# Two stages: the first trains with TensorFlow, the second ships only the light NumPy server,
# so the running container needs a fraction of the memory and disk.
# MovieLens is downloaded and the model trained at build time, so neither is stored in the repo.

FROM python:3.11-slim AS train
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts/download_data.py ./scripts/download_data.py
COPY config.yaml .
RUN python scripts/download_data.py && python -m src.train


FROM python:3.11-slim

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user /home/user/app
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user src ./src
COPY --chown=user config.yaml .
COPY --from=train --chown=user /build/models/mf_weights.npz /build/models/id_mappings.json ./models/
COPY --from=train --chown=user /build/data/raw/ml-100k/u.data /build/data/raw/ml-100k/u.item ./data/raw/ml-100k/

EXPOSE 7860
CMD ["uvicorn", "src.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
