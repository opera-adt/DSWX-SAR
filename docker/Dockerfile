FROM ubuntu:20.04
FROM python:3.9

# Set default UID and GID
ENV USER_ID 1000
ENV GROUP_ID 1000

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

# create user and install latest miniconda
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/requirements_forge.txt /tmp/requirements_forge.txt

RUN set -ex \
 && groupadd -r conda -g $GROUP_ID \
 && useradd -u $USER_ID -r -g conda -d /home/conda -s /bin/bash -m conda \
 && chmod 755 /home/conda \
 && echo 'conda ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda \
 && rm miniconda.sh \
 && touch /opt/conda/conda-meta/pinned \
 && export PATH=/opt/conda/bin:$PATH \
 && conda config --set show_channel_urls True \
 && conda update --all --yes \
 && conda install --yes --file /tmp/requirements.txt \
 && conda install --yes --channel conda-forge --file /tmp/requirements_forge.txt \
 && conda clean -tipy \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/requirements.txt \
 && rm /tmp/requirements_forge.txt

# set default user and workdir
WORKDIR /home/conda
ADD dist/dswx-s1-0.1.0.tar.gz .

USER root
RUN mkdir -p dswx-s1-0.1.0/build
RUN export PATH=/opt/conda/bin:$PATH
WORKDIR /home/conda/dswx-s1-0.1.0
RUN python3 setup.py install
WORKDIR /home/conda

USER conda

ENV PROJ_LIB /opt/conda/share/proj/
ENV GDAL_DIR /opt/conda
ENV PATH $GDAL_DIR/bin:$PATH
ENV GDAL_DATA $GDAL_DIR/share/gdal

ENV PYTHONPATH /home/conda/dswx-s1-0.1.0/src/:$PYTHONPATH
ENV PATH /home/conda/dswx-s1-0.1.0/bin/:$PATH
ENV PATH /home/conda/dswx-s1-0.1.0/src/dswx_sar:$PATH

CMD [ "/bin/bash" ]