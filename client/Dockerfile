FROM python:3.7.3

COPY . ./client

WORKDIR /client

RUN pip install --upgrade pip 

RUN pip install -r client_requirements.txt

ENV TZ Asia/Seoul

EXPOSE 3300

ENTRYPOINT ["python", "cart_client.py"]
