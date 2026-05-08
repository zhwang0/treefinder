# 🌲 **TreeFinder: A US-Scale Benchmark Dataset for Individual Tree Mortality Monitoring Using High-Resolution Aerial Imagery**

> **Accepted to NeurIPS 2025 Datasets & Benchmarks Track**
> - **Paper** is available on OpenReview: https://openreview.net/pdf?id=BH2miB6Smc.
> - **Project page** is available on NeurIPS: https://neurips.cc/virtual/2025/loc/san-diego/poster/121794.
> - **Datasets** are available on [Hugging Face](https://huggingface.co/datasets/zhwang1/TreeFinder) and [Kaggle](https://www.kaggle.com/datasets/zhihaow/tree-finder).


## 🧩 Overview

**TreeFinder** provides a flexible PyTorch-based pipeline for training and evaluating semantic segmentation models on hand-labeled *dead-tree* imagery.  
It supports **custom dataset splits**, **config-driven experiments**, and **benchmarking under multiple generalization scenarios**, aligned with the official *TreeFinder dataset*.



## 🚀 Features

- **Multiple dataset splits** — random (80–10–10), state-based, and scenario-driven (climate / forest type)  
- **Consistent tile loading** — RGB, NIR, NDVI, and no-data mask handling  
- **Augmentations** — flips and 90° rotations for per-tile diversity  
- **Config-driven experiments** — all controlled via a single YAML file  
- **Model zoo** — U-Net, DeepLabV3+, ViT-Seg, SegFormer, Mask2Former, and DOFA  
- **Robust training loop** — masked BCE loss, early stopping, checkpointing, and LR scheduling  
- **Comprehensive evaluation** — IoU, F1, precision, recall, accuracy (micro/macro, all vs. positive-only)  
- **Automated logging** — loss curves, config dumps, and YAML summaries for every run  


## 🧠 Benchmark Models

| Model           | Type               | Description |
|-----------------|--------------------|-------------|
| **U-Net**       | CNN                | [📄Paper](https://arxiv.org/abs/1505.04597) Classic encoder–decoder architecture with skip connections. Trained from scratch as a baseline. |
| **DeepLabV3+**  | CNN                | [📄Paper](https://arxiv.org/abs/1802.02611v3) Employs atrous spatial pyramid pooling with a ResNet-50 backbone. |
| **ViT-Seg**     | Vision Transformer | [📄Paper](https://arxiv.org/abs/2010.11929) Patch-based transformer with a transposed-convolution decoder. |
| **SegFormer**   | Vision Transformer | [📄Paper](https://arxiv.org/abs/2105.15203) Hierarchical transformer with efficient multi-scale feature fusion. |
| **Mask2Former** | Vision Transformer | [📄Paper](https://arxiv.org/abs/2112.01527) Set prediction model using masked attention (Swin-T backbone). |
| **DOFA**        | Foundation Model   | [📄Paper](https://arxiv.org/abs/2403.15356) Multimodal foundation model pretrained on multi-sensor remote-sensing imagery. |

All models are trained with consistent hyperparameters and evaluated under **cross-region**, **cross-climate**, and **cross-forest-type** generalization setups from the *TreeFinder benchmark*.


## 📁 Repository Structure

```text
├── main.py                   # Entrypoint: parse args, load config, run train & eval  
├── configs/                  # YAML experiment configs  
│   ├── debug.yaml  
│   └── benchmark.yaml  
├── data_loader/              # Tile loading & split implementations  
│   ├── __init__.py           # Dispatch get_dataloader(cfg)  
│   ├── utils.py              # load_tile(), augment_tile()  
│   ├── random_split.py  
│   ├── by_state_split.py  
│   ├── by_climate_split.py  
│   └── by_tree_split.py  
├── models/                   # Model factory & builders  
│   ├── __init__.py  
│   ├── unet.py  
│   ├── deeplab.py  
│   ├── vit.py  
│   ├── segformer.py  
│   ├── mask2former.py  
│   └── dofa/                 
├── exps/                     # Training & evaluation loops  
│   ├── train.py  
│   └── evaluate.py  
└── utils/                    # Misc helpers (logging, config I/O, seed control)  
    └── tools.py  
```


## ⚙️ Configuration and ▶️ Quickstart

All experiment parameters (model, optimizer, paths, augmentations, split type) are defined in YAML under `configs/`.  
For example:

```bash
python main.py --config configs/debug.yaml
```


## 📚 Citation

If you use this repository or the **TreeFinder** dataset, please cite:

> **Zhihao Wang**, **Cooper Li**, **Ruichen Wang**, **Lei Ma**, **George Hurtt**, **Xiaowei Jia**, **Gengchen Mai**, **Zhili Li**, **Yiqun Xie**.  
> *TreeFinder: A US-Scale Benchmark Dataset for Individual Tree Mortality Monitoring Using High-Resolution Aerial Imagery.*  
> In *Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)*, 2025.

<!-- ```bibtex
@inproceedings{wang2025treefinder,
  title     = {TreeFinder: A US-Scale Benchmark Dataset for Individual Tree Mortality Monitoring Using High-Resolution Aerial Imagery},
  author    = {Wang, Zhihao and Li, Cooper and Wang, Ruichen and Ma, Lei and Hurtt, George and Jia, Xiaowei and Mai, Gengchen and Li, Zhili and Xie, Yiqun},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), Datasets and Benchmarks Track},
  year      = {2025}
}
``` -->

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Zhihao Wang** — [zhwang1@umd.edu](mailto:zhwang1@umd.edu)
- **Yiqun Xie** — [xie@umd.edu](mailto:xie@umd.edu)
