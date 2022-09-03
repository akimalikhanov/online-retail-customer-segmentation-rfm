FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY ./app  /app
COPY ./data /app/data
COPY ./models /app/models

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]