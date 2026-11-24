# Temel Python image'ımızı belirliyoruz.
FROM python:3.11-slim

# Container içerisindeki çalışma dizinini belirliyoruz.
WORKDIR /app

# Bağımlılık dosyamızı Docker ortamına kopyalıyoruz.
COPY requirements.txt .

# Gerekli Python kütüphanelerini yüklüyoruz.
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI uygulamamızı kopyalıyoruz.
COPY api/ ./api/

# Model ile ilgili Python modüllerimizi kopyalıyoruz.
COPY src/ ./src/

# Container başladığında FastAPI uygulamasını çalıştırıyoruz.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]