FROM python:3.10

WORKDIR /app

# Install transformers library
RUN pip install transformers
RUN pip install tensorflow
RUN pip install datasets
RUN pip install  torch

COPY generated_output_before_fine_tunning.py /app

ENTRYPOINT ["python3", "generated_output_before_fine_tunning.py"]