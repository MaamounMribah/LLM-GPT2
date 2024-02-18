FROM python:3.10.12

WORKDIR /app

# Install transformers library
# RUN pip install transformers
RUN pip install keras_nlp
RUN pip install keras

RUN pip install --upgrade tensorflow

COPY generating_output.py /app
ENTRYPOINT ["python3", "generating_output.py"]