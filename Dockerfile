# Usa un'immagine base di Debian Bullseye con Java 8
FROM openjdk:8-jdk-slim

# Aggiorna e installa i pacchetti di sistema necessari
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    curl \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    libffi-dev \
    && apt-get clean

# Scarica e installa Python 3.10.16
RUN wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tgz && \
    tar -xvzf Python-3.10.16.tgz && \
    cd Python-3.10.16 && \
    ./configure --enable-optimizations && \
    make && make install && \
    cd .. && rm -rf Python-3.10.16 Python-3.10.16.tgz

# Imposta Python 3.10.16 come predefinito
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    python --version

# Installa pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py && rm get-pip.py

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file requirements.txt
COPY requirements.txt /app/requirements.txt

# Installa le dipendenze dal file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Installa MineRL dal repository ufficiale
RUN pip install git+https://github.com/minerllabs/minerl.git

# Clona il nostro repository
RUN git clone https://github.com/MarcoMeg03/MineFIA.git .

# Comando predefinito
CMD ["bash"]
