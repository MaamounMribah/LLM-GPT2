FROM python:3.10

WORKDIR /app

# Install transformers library
RUN pip install transformers
RUN pip install tensorflow
RUN pip install datasets
RUN pip install torch

COPY fine_tune_model.py /app

ENTRYPOINT ["python3", "fine_tune_model.py"]

