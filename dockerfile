FROM ubuntu:24.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Core tools
RUN apt-get update && apt-get install -y \
    curl git python3 python3-pip python3-venv \
    texlive-full \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Install ralph binary (Linux x86-64)
RUN curl -L -o ralph.tar.gz https://github.com/akkeshavan/ralph-releases/releases/latest/download/ralph-LATEST-x86_64-unknown-linux-musl.tar.gz \
    && tar xzf ralph.tar.gz \
    && mv ralph-*/ralph /usr/local/bin/ralph \
    && rm -rf ralph.tar.gz ralph-*/

WORKDIR /research

# Python deps for experiments
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

CMD ["bash"]