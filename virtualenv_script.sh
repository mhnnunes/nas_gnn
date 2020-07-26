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

echo "CACHE: ${CACHE}"

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
conda install -y -c pytorch magma-cuda101 pytorch torchvision && conda clean -ya
conda install h5py
pip install ${CACHE} h5py-cache && \
pip install ${CACHE} torchnet
conda install requests && \
conda install graphviz && conda clean -ya
conda install -y -c menpo opencv3 && conda clean -ya


pip install --verbose ${CACHE} torch-scatter  && \
pip install --verbose ${CACHE} torch-sparse && \
pip install --verbose ${CACHE} torch-cluster && \
pip install --verbose ${CACHE} torch-spline-conv && \
pip install --verbose ${CACHE} torch-geometric && \
pip install --verbose ${CACHE} dgl