FROM oraclelinux:8

LABEL author="OPERA ADT" \
    description="DSWX-SAR final patch2 release R4.2" \
    version="1.2-final-patch2"

RUN yum -y update &&\
    yum -y install curl &&\
    adduser dswx_user

RUN mkdir -p /home/dswx_user/OPERA/DSWX-SAR

RUN chmod -R 755 /home/dswx_user &&\
    chown -R dswx_user:dswx_user /home/dswx_user/OPERA

USER dswx_user

ENV CONDA_PREFIX=/home/dswx_user/miniforge3

# install Miniconda $CONDA_PREFIX
WORKDIR /home/dswx_user
RUN curl -sSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh \
    && bash miniforge.sh -b -p ${CONDA_PREFIX} \
    && rm miniforge.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN ${CONDA_PREFIX}/bin/conda init bash

# copy DSWX-SAR source code and set dswx_user as owner
COPY --chown=dswx_user:dswx_user . /home/dswx_user/OPERA/DSWX-SAR

# create CONDA environment
RUN conda config --add channels conda-forge \
 && conda config --set channel_priority strict

RUN conda create --name "DSWX-SAR" --file /home/dswx_user/OPERA/DSWX-SAR/docker/lock_package.lock &&  conda clean -afy

SHELL ["conda", "run", "-n", "DSWX-SAR", "/bin/bash", "-c"]

WORKDIR /home/dswx_user/OPERA

RUN python -m pip install opencv-python-headless
# installing OPERA DSWX-SAR
RUN python -m pip install ./DSWX-SAR &&\
    echo "conda activate DSWX-SAR" >> /home/dswx_user/.bashrc

WORKDIR /home/dswx_user/scratch

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "DSWX-SAR"]
