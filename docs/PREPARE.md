## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 1.13.1 + cu117

## Installation

We recommend using Anaconda for the installation process:

```bash
# 0. Clone the repository
git clone https://github.com/PeppaWu/SPM
cd SPM/         

# 1. Create and activate 
conda create -n SPM python=3.9 -y
conda activate SPM

# 2. Install PyTorch 2.0 + CUDA 11.7 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
                pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 3. Install basic Python dependencies
pip install -r requirements.txt

# 4. Pre-download three CUDA extension whl files
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/Dao-AILab/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

pip install causal_conv1d-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
        mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
        KNN_CUDA-0.2-py3-none-any.whl

# 7. PointNet++ operation library
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# 8. Compile custom CUDA extensions 
cd ./extensions/chamfer_dist && python setup.py install --user
cd ./extensions/emd          && python setup.py install --user
cd ./mamba                   && python setup.py install
cd ..