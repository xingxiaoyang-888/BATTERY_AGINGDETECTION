FROM node:22-bookworm-slim AS web
WORKDIR /build
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV SODIUM_DATA_DIR=/app/data/workspace SODIUM_MODEL_DIR=/app/artifacts SODIUM_WEB_DIR=/app/web/dist
COPY pyproject.toml README.md ./
COPY sodium_twin/ ./sodium_twin/
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 sodium
COPY --from=web /build/dist /app/web/dist
RUN mkdir -p /app/data/workspace /app/artifacts && chown -R sodium:sodium /app/data /app/artifacts
USER sodium
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v2/health',timeout=3)"
CMD ["python", "-m", "sodium_twin", "--host", "0.0.0.0"]
