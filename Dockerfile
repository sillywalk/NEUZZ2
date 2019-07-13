FROM ubuntu:latest
RUN export DEBIAN_FRONTEND=noninteractive
# Update APT
RUN apt-get update
RUN apt-get install -y \
    python3 \ 
    python3-pip \
    python3-setuptools \
    python3-tk
COPY . /
RUN pip3 install -r requirements.txt
# Run C backend
CMD ["python3", "xtree_plan.py"]
# Run Python backend
CMD ["python3", "xtree_plan.py"]
