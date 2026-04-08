FROM python:3.10-slim

WORKDIR /app

# Only libglib2.0-0 needed for Pillow; libgl1-mesa-glx removed in Debian Trixie
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
