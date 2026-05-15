## RDD: Robust Feature Detector and Descriptor using Deformable Transformer (CVPR 2025) (IMC 2025 Prize Winner)

[Gonglin Chen](https://xtcpete.com/) · [Tianwen Fu](https://twfu.me/) · [Haiwei Chen](https://scholar.google.com/citations?user=LVWRssoAAAAJ&hl=en) · [Wenbin Teng](https://wbteng9526.github.io/) · [Hanyuan Xiao](https://corneliushsiao.github.io/index.html) · [Yajie Zhao](https://ict.usc.edu/about-us/leadership/research-leadership/yajie-zhao/)

[Project Page](https://xtcpete.github.io/rdd/)

## Table of Contents
- [Updates](#updates)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Training](#training)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Updates
[05/14/2026] Retrained RDD and LightGlue after fixing the data leakage issue; more details are available [here](https://github.com/xtcpete/rdd/issues/20).
<table>
  <tr>
    <th></th>
    <th colspan="3">MegaDepth-1500</th>
    <th colspan="3">MegaDepth-View</th>
    <th colspan="3">Air-to-Ground</th>
  </tr>
  <tr>
    <td></td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
    <td>AUC 5&deg</td><td>AUC 10&deg</td><td>AUC 20&deg</td>
  </tr>
  <tr>
    <td>RDD</td>
    <td>51.6</td><td>67.4</td><td>79.6</td>
    <td>52.0</td><td>66.5</td><td>77.8</td>
    <td>56.2</td><td>70.9</td><td>81.0</td>
  </tr>
  <tr>
    <td>RDD + LG</td>
    <td>55.1</td><td>71.2</td><td>82.5</td>
    <td>57.1</td><td>71.5</td><td>81.6</td>
    <td>63.5</td><td>77.7</td><td>86.8</td>
  </tr>
  <tr>
    <td>CVPR version</td>
    <td>48.6</td><td>64.7</td><td>77.2</td>
    <td>38.3</td><td>51.3</td><td>62.3</td>
    <td>49.8</td><td>64.7</td><td>75.8</td>
  </tr>
</table>

[02/11/2026] We updated RDD by replacing the backbone with ConvNeXt for better performance.

[02/11/2026] Air-to-Ground training data is available [here](https://huggingface.co/datasets/xtcpete/air_ground). The aerial images are licensed under CC BY 4.0, and the ground images are sourced from [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/).

[06/06/2025] Evaluation code for ScanNet has been added.

[06/03/2025] We won [4th place](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/xtcpete-4th-place-solution) in the Image Matching Challenge 2025.

[05/16/2025] SfM reconstruction with [COLMAP](https://github.com/colmap/colmap.git) has been added. We provide a ready-to-use [notebook](./demo_sfm.ipynb) for a simple example. Code adapted from [hloc](https://github.com/cvg/Hierarchical-Localization.git).

[05/12/2025] Training code and new weights released.

## Installation

```bash
git clone --recursive https://github.com/xtcpete/rdd
cd rdd

# Create conda env
conda create -n rdd python=3.10 pip
conda activate rdd

# Install CUDA 
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
# Install torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# Install all dependencies
pip install -r requirements.txt
# Compile custom operations.
# You don't have to compile them to run RDD, but it is recommended for better performance.
cd ./RDD/models/ops
pip install -e .
```

We provide the [download link](https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA?usp=sharing) for:
  - the MegaDepth-1500 test set
  - the MegaDepth-View test set
  - the Air-to-Ground test set
  - the ScanNet-1500 test set
  - two pretrained models: RDD and LightGlue for matching RDD features

Create the `data` folder and unzip the downloaded test data into it.

Create the `weights` folder, add the weights to it, and you are ready to go.

## Usage
For your convenience, we provide a ready-to-use [notebook](./demo_matching.ipynb) with examples.

### Inference

```python
from RDD.RDD import build

RDD_model = build()

output = RDD_model.extract(torch.randn(1, 3, 480, 640))
```

### Evaluation

Please note that due to different GPU architectures and the stochastic nature of RANSAC, you may observe slightly different results; however, they should be very close to those reported in the paper. To reproduce the numbers in the paper, use the v1 weights instead.

Results can be visualized by passing `--plot`. The CVPR-version model can be evaluated by passing `--config` with the corresponding weights.

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

**ScanNet-1500**

```bash
# Sparse matching
python ./benchmarks/scannet_1500.py

# Dense matching
python ./benchmarks/scannet_1500.py --method dense

# LightGlue
python ./benchmarks/scannet_1500.py --method lightglue
```

### Training

1. Download the MegaDepth dataset using [download.sh](./data/megadepth/download.sh) and the megadepth_indices from [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets). Then the MegaDepth root folder should look like the following:
```bash
./data/megadepth/megadepth_indices # indices
./data/megadepth/depth_undistorted # depth maps
./data/megadepth/Undistorted_SfM # images and poses
./data/megadepth/scene_info # indices for training LightGlue
```
2. Download the Air-to-Ground training data from [here](https://huggingface.co/datasets/xtcpete/air_ground).
3. Then you can train RDD in two steps: first, the descriptor
```bash
python -m training.train 

# You can train the RDD CVPR version with
python -m training.train --config ./configs/cvpr.yaml --no-crop
```
and then the detector
```bash
python -m training.train --detector_from /path/to/descriptor.pth --train_detector --training_res 480

# You can train the RDD CVPR version with
python -m training.train --detector_from /path/to/descriptor.pth --train_detector --training_res 480 --config ./configs/cvpr.yaml --no-crop
```

## Citation
```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Gonglin and Fu, Tianwen and Chen, Haiwei and Teng, Wenbin and Xiao, Hanyuan and Zhao, Yajie},
    title     = {RDD: Robust Feature Detector and Descriptor using Deformable Transformer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6394-6403}
}
```


## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements

We thank the authors of these great repositories: [ALIKE](https://github.com/Shiaoming/ALIKE), [LoFTR](https://github.com/zju3dv/LoFTR), [DeDoDe](https://github.com/Parskatt/DeDoDe), [XFeat](https://github.com/verlab/accelerated_features), [LightGlue](https://github.com/cvg/LightGlue), [Kornia](https://github.com/kornia/kornia), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), along with many other inspiring works from the community.

LightGlue is trained with [Glue Factory](https://github.com/cvg/glue-factory).

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0075. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government. We would like to thank Yayue Chen for her help with visualization.
