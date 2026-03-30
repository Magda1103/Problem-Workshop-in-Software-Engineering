FROM python:3.11

# Installation of system dependencies for OpenCV and video.
# Update on packages for newer Debian versions.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setting the working directory
WORKDIR /app

# Copying and installing Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the code
COPY . .

CMD ["python", "main.py"]