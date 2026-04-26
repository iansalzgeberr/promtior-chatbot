# Dockerfile para el backend LangServe
# Usamos Python 3.11 slim para mantener la imagen liviana
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para FAISS y requests
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (aprovecha cache de Docker)
# Si solo cambia el codigo pero no las dependencias, no reinstala todo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto en el que corre LangServe
EXPOSE 8000

# Ejecutar el servidor al iniciar el contenedor
# --host 0.0.0.0 permite conexiones externas (necesario en Docker)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
