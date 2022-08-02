FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

COPY . /code/

WORKDIR /code/

RUN apt-get update && apt-get install -y \
    curl  \
    git  \
    pkg-config  \
    cmake &&\
    rm -rf /var/lib/apt/lists/* \
    apt install software-properties-common \
    apt install graphviz


RUN python -m pip install jupyterlab
RUN python -m pip install poetry
RUN poetry config virtualenvs.in-project false
RUN make install-deps

CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]