FROM pytorch/pytorch

RUN apt update && apt install -y\ 
    python3 \
    python3-pip
RUN pip3 install flask msgpack pillow 

WORKDIR /app
COPY ./gender.pt .
COPY ./inference.py .
COPY ./model.py .
COPY ./genderfilter.py .

CMD python3 inference.py