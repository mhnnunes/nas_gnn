wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda install conda-build
conda create -y --name py37 python=3.7.4 && \
conda clean -ya
conda activate py37
conda install pip
conda install -y -c pytorch magma-cuda101 pytorch torchvision && conda clean -ya
conda install h5py
pip install --cache-dir /srv/mhnnunes/pipcache h5py-cache && \
pip install --cache-dir /srv/mhnnunes/pipcache torchnet
conda install requests && \
conda install graphviz && conda clean -ya
conda install -y -c menpo opencv3 && conda clean -ya


pip install --verbose --cache-dir /srv/mhnnunes/pipcache torch-scatter  && \
pip install --verbose --cache-dir /srv/mhnnunes/pipcache torch-sparse && \
pip install --verbose --cache-dir /srv/mhnnunes/pipcache torch-cluster && \
pip install --verbose --cache-dir /srv/mhnnunes/pipcache torch-spline-conv && \
pip install --verbose --cache-dir /srv/mhnnunes/pipcache torch-geometric && \
pip install --verbose --cache-dir /srv/mhnnunes/pipcache dgl