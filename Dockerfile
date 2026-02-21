# Imagen base estable con CUDA 12.1 y cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/workspace/.hf_cache
ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
ENV HF_DATASETS_CACHE=/workspace/.hf_cache

# Dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y \
    git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Instalar deps Python (cacheable)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiamos SOLO código (no data) para mantener cache
COPY configs/ ./configs/
COPY src/ ./src/
COPY run_all.py ./run_all.py
COPY validate_data.py ./validate_data.py

# Directorios para montajes
RUN mkdir -p /workspace/data /workspace/experiments /workspace/.hf_cache

# Comando por defecto: validar y correr todo
CMD ["bash", "-lc", "python validate_data.py && python run_all.py"]