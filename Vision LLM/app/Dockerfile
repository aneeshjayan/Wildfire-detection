FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install Python dependencies in chunks
RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir pillow

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]

