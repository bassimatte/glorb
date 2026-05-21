FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

ENV PORT=10000
EXPOSE ${PORT}

CMD gunicorn server:app --bind 0.0.0.0:${PORT} --timeout 300 --workers 1
