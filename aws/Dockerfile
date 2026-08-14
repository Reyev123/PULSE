FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8050 \
    OLLAMA_URL=http://127.0.0.1:11434

WORKDIR /app

# Runtime-only image: training data is intentionally excluded.
COPY GUI/requirements.txt /app/GUI/requirements.txt
RUN pip install --no-cache-dir -r /app/GUI/requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY GUI /app/GUI
COPY models/rhythm_cnn.py models/rhythm_infer.py models/rhythm_cnn.pt /app/models/
COPY tools/render_single_lead.py /app/tools/render_single_lead.py

EXPOSE 8050
CMD ["python", "GUI/app.py"]
