# Dead‐Tree Detection & Segmentation Benchmark

A flexible PyTorch pipeline for building, training, and evaluating remote‐sensing segmentation models on hand‐labeled “dead tree” tiles across the CONUS. Designed for the NeurIPS 2025 Datasets & Benchmarks track, it supports:

- **Multiple dataset splits** (random 80–10–10, state-based, custom scenarios)  
- **Consistent tile loading** (RGB, NIR, NDVI bands + no-data mask)  
- **Data augmentations** (flips + per-tile 90° rotations)  
- **Config-driven experiments** via a single `configs.yaml`  
- **Model zoo**: UNet, ViT‐based, SegFormer, DeepLabV3, and TF-ported DeepLabV3+  
- **Training loop** with masked BCE loss, early stopping, checkpointing, LR schedulers  
- **Evaluation** of pixel‐level IoU, F1, precision, recall, accuracy under micro/macro averaging and “all” vs “positive_only” scenarios  
- **Logging & results**: automatic log files, loss-curve plots, and YAML summaries  

---


## 🔍 Benchmark Models

We benchmark six semantic segmentation models to establish strong baselines for individual-level tree mortality mapping:

| Model           | Type                 | Description |
|-----------------|----------------------|-------------|
| **U-Net**       | CNN                  | [📄Paper](https://arxiv.org/abs/1505.04597) Classic encoder–decoder architecture with skip connections. Trained from scratch as a baseline without pretraining. 
| **DeepLabV3+**  | CNN                  | [📄Paper](https://arxiv.org/abs/1802.02611v3) Uses atrous spatial pyramid pooling with a ResNet-50 backbone. Pretrained on ImageNet.
| **ViT-Seg**     | Vision Transformer   | [📄Paper](https://arxiv.org/abs/2010.11929) Patch-based transformer with a lightweight transposed-convolution decoder. Pretrained on ImageNet.
| **SegFormer**   | Vision Transformer   | [📄Paper](https://arxiv.org/abs/2105.15203) Hierarchical transformer with multi-scale feature encoding and an efficient decoder. Pretrained on ADE20K.
| **Mask2Former** | Vision Transformer   | [📄Paper](https://arxiv.org/abs/2112.01527) Set prediction model using masked attention and a Swin-Tiny backbone. Pretrained on ADE20K.
| **DOFA**        | Foundation Model     | [📄Paper](https://arxiv.org/abs/2403.15356) Multimodal foundation model pretrained on diverse remote sensing imagery (e.g., Sentinel, NAIP). Adapted for semantic segmentation.

Each model is trained using consistent hyperparameters and evaluated across multiple generalization scenarios (e.g., cross-region, cross-climate, and cross-forest-type) using TreeFinder.



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
└── utils/                    # Misc helpers (logging, config I/O, seed)  
    └── tools.py 
```

## 🛠️ Configuration
All hyperparameters and paths live in configs/*.yaml. All test scenarios are also included in configs/.


## ▶️ Quickstart
```
python main.py --config configs/debug.yaml
```

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Zhihao Wang** — [zhwang1@umd.edu](mailto:zhwang1@umd.edu)
- **Yiqun Xie** — [xie@umd.edu](mailto:xie@umd.edu)
