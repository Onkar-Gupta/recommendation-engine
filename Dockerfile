FROM python:3.11-slim

# Model and data are mounted at run time (see `make docker-run`), not baked in:
# no training or MovieLens download at build time, and one image serves any trained model.
# The server runs on NumPy and pandas only (requirements-serve.txt), so the image stays small.
RUN useradd --create-home app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY config.yaml .
RUN mkdir logs && chown app logs

USER app
EXPOSE 8000
CMD ["uvicorn", "src.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
