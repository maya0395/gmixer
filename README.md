# G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval

> **[CVPR 2026]** Official repository for G-MIXER

 
<!-- PAPER FIGURE HERE — 아래 주석 해제 후 assets/framework.png 추가 -->
<!-- ![G-MIXER](assets/framework.png) -->

---

## 📖 Abstract

Composed Image Retrieval (CIR) aims to retrieve target images by integrating a reference image with a corresponding modification text. CIR requires jointly considering the explicit semantics specified in the query and the implicit semantics embedded within its bi-modal composition. Recent training-free zero-shot CIR (ZS-CIR) methods leverage Multimodal Large Language Models (MLLMs) to generate detailed target descriptions, converting the implicit information into explicit textual expressions. However, these methods rely heavily on the textual modality and fail to capture the fuzzy retrieval nature that requires considering diverse combinations of candidates. This leads to reduced diversity and accuracy in retrieval results. To address this limitation, we propose a novel training-free method, Geodesic Mixup-based Implicit semantic eXpansion and Explicit semantic Re-ranking for ZS-CIR (G-MIXER). G-MIXER constructs composed query features that reflect the implicit semantics of reference image-text pairs through geodesic mixup over a range of mixup ratios, and builds a diverse candidate set. The generated candidates are then re-ranked using explicit semantics derived from MLLMs, improving both retrieval diversity and accuracy. Our proposed G-MIXER achieves state-of-the-art performance across multiple ZS-CIR benchmarks, effectively handling both implicit and explicit semantics without additional training.


## 🌟 Key Features

- **Geodesic Mixup**: Interpolates between reference image and text embeddings along the geodesic path on the hypersphere, generating semantically meaningful intermediate representations for implicit semantic expansion.
- **Implicit Semantic Expansion**: Augments the query embedding space to capture a broader range of semantically consistent target images.
- **Explicit Semantic Re-ranking**: A post-retrieval module that re-scores retrieved candidates using fine-grained semantic alignment with the modification text.
- **Training-Free**: Operates entirely in a zero-shot manner, leveraging pre-trained CLIP/OpenCLIP without any task-specific fine-tuning.

---

## 🔧 Setup

### Environment
```bash
conda create -n gmixer python=3.9 -y
conda activate gmixer
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch transformers tqdm termcolor pandas
pip install git+https://github.com/openai/CLIP.git
```

### Datasets

G-MIXER is evaluated on the following standard ZS-CIR benchmarks.

#### FashionIQ

Download the FashionIQ dataset from the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq). The expected folder structure is:
```text
├── FASHIONIQ
│   ├── captions
│   │   ├── cap.dress.[train | val | test].json
│   │   ├── cap.toptee.[train | val | test].json
│   │   └── cap.shirt.[train | val | test].json
│   ├── image_splits
│   │   ├── split.dress.[train | val | test].json
│   │   ├── split.toptee.[train | val | test].json
│   │   └── split.shirt.[train | val | test].json
│   └── images
│       └── [*.jpg]
```

#### CIRR

Download the CIRR dataset from the [official repository](https://github.com/Cuberick-Orion/CIRR). The expected folder structure is:
```text
├── CIRR
│   ├── train
│   │   └── [0 | 1 | 2 | ...]
│   │       └── [*.png]
│   ├── dev
│   │   └── [*.png]
│   ├── test1
│   │   └── [*.png]
│   └── cirr
│       ├── captions
│       │   └── cap.rc2.[train | val | test1].json
│       └── image_splits
│           └── split.rc2.[train | val | test1].json
```

#### CIRCO

Download the CIRCO dataset from the [official repository](https://github.com/miccunifi/CIRCO). The expected folder structure is:
```text
├── CIRCO
│   ├── annotations
│   │   └── [val | test].json
│   └── COCO2017_unlabeled
│       ├── annotations
│       │   └── image_info_unlabeled2017.json
│       └── unlabeled2017
│           └── [*.jpg]
```

---

## 🚀 Running G-MIXER

 

### Key Arguments

| Argument | Description |
|---|---|
| `--dataset` | Dataset to evaluate (`fashioniq_dress`, `fashioniq_shirt`, `fashioniq_toptee`, `cirr`, `circo`) |
| `--split` | Evaluation split: `val` for metrics, `test` for test submission |
| `--datapath` | Path to the dataset root folder |
| `--clip_model_name` | OpenCLIP backbone (e.g., `ViT-B-32`, `ViT-L-14`, `ViT-bigG-14`) |
| `--geodesic_ratio_list` | Number of geodesic interpolation steps for semantic expansion |
| `--top-k` | Top-K candidates used in the explicit re-ranking stage |
| `--json_path` | new caption file path |



---
 

## 🗂️ Repository Structure
```
G-MIXER/
├── src/
│   ├── utils.py
│   ├── datasets.py

│   ├── o_dress.ipynb
│   ├── o_toptee.ipynb
│   ├── o_shirt.ipynb               

├── assets/
│   └── framework.png        # Architecture figure

├── LICENSE
└── README.md
```

---

## 📝 Citation

If you find this work helpful, please consider citing:
```bibtex
@InProceedings{GMIXER_CVPR2026,
    author    = {Jiyoung Lim, Heejae Yang, and Jee-Hyong Lee},
    title     = {G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
}
```

---

## 🤝 Acknowledgements

This codebase builds upon the following outstanding works. We sincerely thank their authors for open-sourcing their code:

- [CIReVL](https://github.com/ExplainableML/Vision_by_Language) (ICLR 2024) — our baseline code is adapted from here
- [OSrCIR](https://github.com/Pter61/osrcir) (CVPR 2025) — our baseline code is adapted from here
- [SEARLE](https://github.com/miccunifi/SEARLE) 
- [CIRCO](https://github.com/miccunifi/CIRCO)
- [CIRR](https://github.com/Cuberick-Orion/CIRR)

---

## 📧 Contact

For questions, feel free to open an issue or reach out to maya0707@skku.edu.
