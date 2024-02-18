FROM python:3.10

WORKDIR /app

# Install transformers library
RUN pip install transformers
RUN pip install tensorflow
RUN pip install datasets

COPY generated_output before_fine _tunning.py /app

ENTRYPOINT ["python3", "generated_output before_fine _tunning.py"]