FROM python:3.7.3

COPY . ./server

WORKDIR /server

RUN pip install --upgrade pip 

RUN . env/bin/activate

RUN pip install -r server_requirements.txt

ENV TZ Asia/Seoul

EXPOSE 3300

ENTRYPOINT ["python", "cart_server.py"]
