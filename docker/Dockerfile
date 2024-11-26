# Start from a CUDA base image (e.g., CUDA 12.2 runtime with Ubuntu 20.04)
FROM rapidsai/miniforge-cuda:cuda12.5.1-base-ubuntu20.04-py3.12

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 

# Update pip to the latest version
RUN python -m pip install --upgrade pip

# Install Poetry for dependency management
RUN python -m pip install poetry

# Set the working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of the application files into the container
COPY . /app/

# Set entrypoint for running the application
ENTRYPOINT ["poetry", "run", "python", "task.py"]

# Set CMD with default arguments
CMD ["--config_path", "./config/config.yaml"]
