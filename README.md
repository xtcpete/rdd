## RDD: Robust Feature Detector and Descriptor using Deformable Transformer (CVPR 2025)
[Gonglin Chen](https://xtcpete.com/) · [Tianwen Fu](https://twfu.me/) · [Haiwei Chen](https://scholar.google.com/citations?user=LVWRssoAAAAJ&hl=en) · [Wenbin Teng](https://wbteng9526.github.io/) · [Hanyuan Xiao](https://corneliushsiao.github.io/index.html) · [Yajie Zhao](https://ict.usc.edu/about-us/leadership/research-leadership/yajie-zhao/)

[Project Page](https://xtcpete.github.io/rdd/) 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Training](#training) - Coming soon
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

```bash
git clone --recursive https://github.com/xtcpete/rdd
cd RDD

# Create conda env
conda create -n rdd python=3.8 pip
conda activate rdd

# Install CUDA 
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
# Install torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# Install all dependencies
pip install -r requirements.txt
# Compile custom operations
cd ./RDD/models/ops
sh make.sh
```

We provide the [download link](https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA?usp=sharing) to:
  - the MegaDepth-1500 test set
  - the MegaDepth-View test set
  - the Air-to-Ground test set
  - 2 pretrained models, RDD and LightGlue for matching RDD

Create and unzip downloaded test data to the `data` folder.

Create and add weights to the `weights` folder and you are ready to go.

## Usage
For your convenience, we provide a ready-to-use [notebook](./Demo.ipynb) for some examples.

### Inference

```python
from RDD.RDD import build

RDD_model = build()

output = RDD_model.extract(torch.randn(1, 3, 480, 640))
```

### Evaluation

Please note that due to the different GPU architectures and the stochastic nature of RANSAC, you may observe slightly different results; however, they should be very close to those reported in the paper.

**MegaDepth-1500**

```bash
# Sparse matching
python ./benchmarks/mega_1500.py

# Dense matching
python ./benchmarks/mega_1500.py --method dense

# LightGlue
python ./benchmarks/mega_1500.py --method lightglue
```

**MegaDepth-View**

```bash
# Sparse matching
python ./benchmarks/mega_view.py

# Dense matching
python ./benchmarks/mega_view.py --method dense

# LightGlue
python ./benchmarks/mega_view.py --method lightglue
```

**Air-to-Ground**

```bash
# Sparse matching
python ./benchmarks/air_ground.py

# Dense matching
python ./benchmarks/air_ground.py --method dense

# LightGlue
python ./benchmarks/air_ground.py --method lightglue
```

### Training

I am working on cleaning up the code and dataset. Stay tuned!

## Citation

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements

We thank these great repositories: [ALIKE](https://github.com/Shiaoming/ALIKE), [LoFTR](https://github.com/zju3dv/LoFTR), [DeDoDe](https://github.com/Parskatt/DeDoDe), [XFeat](https://github.com/verlab/accelerated_features), [LightGlue](https://github.com/cvg/LightGlue), [Kornia](https://github.com/kornia/kornia), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and many other inspiring works in the community.

LightGlue is trained with [Glue Factory](https://github.com/cvg/glue-factory).

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0075. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government. We would like to thank Yayue Chen for her help with visualization.