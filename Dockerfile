FROM python:3.8

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

WORKDIR /app
COPY . /app
RUN wget https://storage.googleapis.com/twitter-em-roberta-models/twitter-models.zip


RUN unzip twitter-models.zip

CMD ["uvicorn","app.main:app","--host",,"--port","80"]
