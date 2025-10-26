FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Symlink de compatibilité si nécessaire
RUN if [ ! -f /usr/lib/x86_64-linux-gnu/libgomp-a34b3233.so.1.0.0 ] && [ -f /usr/lib/x86_64-linux-gnu/libgomp.so.1 ]; then \
      ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp-a34b3233.so.1.0.0 ; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install --no-cache-dir -r /tmp/requirements.txt
