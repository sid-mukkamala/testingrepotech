FROM python:3.7-slim

COPY . /

WORKDIR /

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]