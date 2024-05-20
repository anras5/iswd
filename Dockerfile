FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y glpk-utils graphviz\
    && rm -rf /var/lib/apt/lists/*

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
