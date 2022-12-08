FROM python:3.10.1-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inference.py .
COPY main.py .
COPY classifier.pkl .
COPY naivebayes.py .
COPY vectorizer.pkl .
EXPOSE 8501

CMD ["streamlit", "run", "main.py"]