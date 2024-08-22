FROM python:3.10

WORKDIR /app


COPY requirements.txt requirements.txt


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Exposer le port 8000 pour Flask
EXPOSE 5000


RUN pip install gunicorn


CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
