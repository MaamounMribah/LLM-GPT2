FROM python:3.10

WORKDIR /app

# Install transformers library
RUN pip install transformers[torch]
RUN pip install datasets
RUN pip install tensorflow

RUN pip install torch

COPY fine_tune_model_output.py /app

ENTRYPOINT ["python3", "fine_tune_model_output.py"]
