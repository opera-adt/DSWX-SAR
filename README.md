# DSWX-SAR
Dynamic Surface Water Extent from Synthetic Aperture Radar

## Installation

### Download the Source Code
Download the source code and change working directory to cloned repository:

```bash
git clone https://github.com/opera-adt/DSWX-SAR.git
```

### Standard Installation
Install dependencies (installation via conda is recommended):
```bash
conda install --file docker/requirements.txt
conda install -c conda-forge --file docker/requirements_forge.txt
```

Install via setup.py:

```bash
python setup.py install
python setup.py clean
```

Note: Installation via pip is not currently recommended due to an
issue with the osgeo and gdal dependency.


OR update environment path to run DSWX-SAR:

```bash
export DSWX_SAR_HOME=$PWD
export PYTHONPATH=${PYTHONPATH}:${DSWX_SAR_HOME}/src
export PATH=${PATH}:${DSWX_SAR_HOME}/bin
```

Process data sets; use a runconfig file to specify the location
of the dataset, the output directory, parameters, etc.

```bash
dswx_s1.py <path to runconfig file>
```
Note: Only Sentinel-1 data is currently supported. 

A default runconfig file can be found: `DSWX-SAR > src > dswx_sar > defaults > dswx_s1.yaml`.
This file can be copied and modified for your needs.
Note: The runconfig must meet this schema: `DSWX-SAR > src > dswx_sar > schemas > dswx_s1.yaml`.


### Alternate Installation: Docker Image

Skip the standard installation process above.

Then, from inside the cloned repository, build the Docker image:
(This will automatically run the workflow tests.)

```bash
./build_docker_image.sh
```

Load the Docker container image onto your computer:

```bash
docker load -i docker/dockerimg_dswx_s1_calval_0.4.tar
```

See DSWx-SAR Science Algorithm Software (SAS) User Guide for instructions on processing via Docker.
