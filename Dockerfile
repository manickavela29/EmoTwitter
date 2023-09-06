FROM python:3.8

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

WORKDIR /app
COPY . .
#RUN wget https://storage.googleapis.com/twitter-em-roberta-models/model.zip
RUN unzip model.zip

EXPOSE 5000

CMD [ "flask","run","--host", "0.0.0.0:5000"]