FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY audio_transcriber/ /app/audio_transcriber/
COPY rag_service/ /app/rag_service/
CMD [ "python", "rag_service/rag_app.py" ]