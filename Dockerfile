FROM tensorflow/tensorflow:2.12.0-gpu

# 1) Paquetes del sistema
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    git \
    build-essential \
    wget \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# 2) Actualizar pip/setuptools/wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3) Clonar TF Models (opcional: fijar commit para estabilidad)
RUN git clone https://github.com/tensorflow/models.git /models

WORKDIR /models/research

# 4) Compilar protos
RUN protoc object_detection/protos/*.proto --python_out=.

# 5) Instalar la API de detección TF2
RUN cp object_detection/packages/tf2/setup.py . && python3 -m pip install .

# 6) Variables de entorno para resolver imports
ENV PYTHONPATH=/models/research:/models/research/slim:${PYTHONPATH}
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 7) Requisitos adicionales del proyecto
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# 8) Directorio de trabajo para tu código/datos
WORKDIR /workspace
