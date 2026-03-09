FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/CosyVoice

RUN apt-get update && \
    apt-get install -y git git-lfs ffmpeg libsndfile1 sox && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/CosyVoice

RUN git lfs install && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r runtime/python/vllm_adapter/requirements.txt

EXPOSE 8001

CMD ["uvicorn", "runtime.python.vllm_adapter.app:app", "--host", "0.0.0.0", "--port", "8001"]
