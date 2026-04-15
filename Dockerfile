FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest build twine

COPY . .

RUN pip install --no-cache-dir .

CMD ["pytest"]