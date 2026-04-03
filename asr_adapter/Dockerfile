FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx python-multipart

COPY main.py .

EXPOSE 4000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
