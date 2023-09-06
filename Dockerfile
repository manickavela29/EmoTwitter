FROM python:3.8

COPY requirements.txt /tmp/

RUN pip install --requirement /tmp/requirements.txt

WORKDIR /app
RUN wget "https://drive.google.com/uc?id=12yRZeaunIvKmm8euxchqH_Y12asn3taC&export=download"
#RUN wget f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emotion/mapping.txt"
COPY ModelServingTwitterEmotion.zip ModelServingTwitterEmotion.zip
RUN ls 
RUN unzip ModelServingTwitterEmotion.zip -d /app/model/

CMD ["bash"]