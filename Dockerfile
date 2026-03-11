FROM python:3.11-slim

# Instalacja zależności systemowych dla OpenCV i wideo
# Zaktualizowano pakiety dla nowszych wersji Debiana
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie i instalacja bibliotek Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie reszty kodu
COPY . .

CMD ["python", "main.py"]