# Temel Python image'ımızı belirliyoruz.
FROM python:3.11-slim

# Çalışma dizinimizi belirliyoruz.
WORKDIR /app

# Bağımlılık dosyasını kopyalıyoruz.
COPY requirements.txt .

# Kütüphaneleri yüklüyoruz.
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uygulamamızı kopyalıyoruz.
COPY frontend/ ./frontend/

# Container başladığında Streamlit uygulamasını çalıştırıyoruz.
CMD ["python", "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]