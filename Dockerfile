# Użyj obrazu bazowego z Pythona
FROM python:3.9-slim

# Ustaw zmienną środowiskową, aby Python nie buforował wyjścia
ENV PYTHONUNBUFFERED=1

# Utwórz katalog roboczy
WORKDIR /app

# Skopiuj plik requirements.txt do katalogu roboczego
COPY requirements.txt .

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj pliki aplikacji do katalogu roboczego
COPY app.py .
COPY model.pkl .

# Expose port 5000
EXPOSE 5000

# Uruchom aplikację
CMD ["python", "app.py"]