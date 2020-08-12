#!/bin/bash

# This script creates a virtual environment using Miniconda and 
# installs the dependencies necessary to run the code in this 
# repository. This script considers that the computer is using
# CUDA 10.1. If this is not your CUDA version, change
# the lines with comments highlighted below

CACHE=""
if [[ $1 != "" ]]; then
	CACHE="--cache-dir ${1}"
fi

echo "USING CACHE: ${CACHE}"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda install conda-build
conda create -y --name py37 python=3.7.4 && \
conda clean -ya
conda activate py37
conda install pip
# IMPORTANT: if your computer is not using CUDA 10.1 the line below 
# has to contain a different version of magma-cuda.
conda install -y -c pytorch=1.3.1 magma-cuda101=2.5.1 torchvision=0.4.2 && \
conda clean -ya
conda install h5py=2.9.0
pip install --verbose ${CACHE} h5py-cache && \
pip install --verbose ${CACHE} torchnet
conda install requests=2.22.0 && \
conda install graphviz=2.40.1 && conda clean -ya
conda install -y -c menpo opencv3 && conda clean -ya


pip install --verbose ${CACHE} torch-scatter==1.4.0  && \
pip install --verbose ${CACHE} torch-sparse==0.4.3 && \
pip install --verbose ${CACHE} torch-cluster==1.4.5 && \
pip install --verbose ${CACHE} torch-spline-conv==1.1.1 && \
pip install --verbose ${CACHE} torch-geometric==1.3.2 && \
pip install --verbose ${CACHE} dgl==0.4.1