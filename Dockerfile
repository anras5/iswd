FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]