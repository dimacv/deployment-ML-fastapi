FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY cloudpickle.pkl cloudpickle.pkl

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
