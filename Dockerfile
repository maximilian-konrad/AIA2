# Use CUDA 12.4 base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/miniconda && \
    rm /miniconda.sh && \
    apt-get clean && \
    echo 'export PATH=/opt/miniconda/bin:$PATH' >> ~/.bashrc

# Set Conda environment variables
ENV PATH="/opt/miniconda/bin:$PATH"
ENV CONDA_ALWAYS_YES="true"

# Set working directory
WORKDIR /app

# Copy necessary files
COPY . /app/

# Create conda environment from the yml file
RUN conda env create -f env.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "aia", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.4
RUN conda run -n aia pip install torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124 \
    torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install additional packages if needed
RUN conda run -n aia pip install flask gunicorn

# Make port 5000 available for the app
EXPOSE 5000

# Set entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["app"]
