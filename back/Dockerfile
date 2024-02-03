
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY data.py .
COPY model.py .
COPY run_back.py .
COPY datas/datas.csv datas/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["python", "run_back.py"]