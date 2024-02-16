FROM python:3.10

WORKDIR /app

# Install transformers library
RUN pip install transformers
RUN pip install tensorflow
RUN pip install datasets

COPY preprocess_data.py /app

ENTRYPOINT ["python3", "preprocess_data.py"]

