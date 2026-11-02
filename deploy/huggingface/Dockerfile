# Hugging Face Space image. Build context is the repo root (or the staged copy made by push.ps1).
# MovieLens is downloaded and the model trained at build time, so neither is stored in the repo.
FROM python:3.11-slim

# Spaces run containers as user 1000.
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user /home/user/app
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user scripts/download_data.py ./scripts/download_data.py
COPY --chown=user config.yaml .
RUN python scripts/download_data.py && python -m src.train

EXPOSE 7860
CMD ["uvicorn", "src.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
