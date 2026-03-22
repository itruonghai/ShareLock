# ShareLock: Ultra-Lightweight CLIP-like Vision-Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2410.07173-b31b1b.svg)](https://arxiv.org/abs/2410.07173)
[![Project Page](https://img.shields.io/badge/Project_Page-ShareLock-b31b1b)]([jonaruthardt.github.io/projects/ShareLock/](https://jonaruthardt.github.io/projects/ShareLock/))
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This repository provides the official implementation of *ShareLock*, an ultra-lightweight CLIP-like vision-language model, introduced in the paper:  
**"Better Language Models Exhibit Higher Visual Alignment"**  
*[Jona Ruthardt](https://jonaruthardt.github.io/), [Gertjan J. Burghouts](https://gertjanburghouts.github.io), [Serge Belongie](https://sergebelongie.github.io/), [Yuki M. Asano](yukimasano.github.io/)*  
Published in **Transactions on Machine Learning Research (TMLR) 01/2026**

📄 **[Read the Paper on arXiv](https://arxiv.org/pdf/2410.07173)**  
🤗 **[Model Checkpoints on Hugging Face](https://huggingface.co/FunAILab/ShareLock)**  
🌐 **[More Information on our Project Page](https://jonaruthardt.github.io/projects/ShareLock/)**  

---

## 🧪 Overview

![Workflow Diagram](DiagramShareLock.png)

**ShareLock** is a straightforward and efficient approach to building vision-language models. By leveraging frozen features from strong unimodal vision and language models, it achieves competitive multimodal performance with minimal computational resources. Key highlights include:  

- **Data Efficiency**: Trained on just 563k image-caption pairs, ShareLock achieves 51% zero-shot accuracy on ImageNet.  
- **Cost Efficiency**: Training requires only 1 GPU hour (10 hours including feature precomputation).  
- **Competitive Results**: Outperforms existing models in low-data regimes while maintaining scalability.

---

## 🚀 Features

- **Ultra-Lightweight:** Minimal training time with competitive results.
- **Pretrained Backbone:** Leverages strong, frozen unimodal features.
- **Low Resource Requirement:** Trainable with only one GPU in hours.
- **Zero-Shot Capabilities:** Effective on ImageNet and beyond.
- **CLIP-like VLM:** apply common refinement techniques (e.g., prompt tuning, LLM-based descriptions, etc.)

---

## 🛠️ Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/JonaRuthardt/ShareLock.git
    cd ShareLock
    ```

2. **Set up a Python Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📦 Usage

### Quick Start: Train + Evaluate

The `run_experiment.sh` script handles the full pipeline (training → evaluation) in one command:

```bash
# Train and auto-evaluate on ImageNet-1k zero-shot
./run_experiment.sh --config configs/cc3m_llava_config.yaml --gpu 0

# Use a specific GPU
./run_experiment.sh --config configs/cc3m_llava_config.yaml --gpu 1

# Evaluate an existing checkpoint (no training)
./run_experiment.sh --eval-only checkpoints/ShareLock-CC3M.ckpt \
    --config configs/cc3m_llava_config.yaml

# Train with 2 GPUs (DDP) — call train.py directly
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/cc3m_nv_former.yaml
```

Monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/ --port 6006
```

### Step-by-Step

0. **Download Datasets**:
    Training and validation of the model requires the presence of paired image-caption data. Popular small-scale datasets include CC3M, CC12M and YFCC15M and can be downloaded in webdataset format using the [img2dataset](https://github.com/rom1504/img2dataset/) library.

1. **Precompute Features**:
    Use pretrained models to extract vision and text embeddings:
    ```bash
    python precompute_image_features_hf.py  # HuggingFace datasets for classification tasks (test)
    python precompute_image_features_wds.py  # Image-caption datasets in webdataset format (train/val)
    python precompute_language_features.py   # JSON file with captions for each uid in image dataset
    ```
    The dataset and backbone model can be configured via command line arguments. The presence of a JSON file with image *uids* as keys and corresponding captions as values is assumed.

2. **Train the Projection Network**:
    ```bash
    python train.py --config configs/cc3m_llava_config.yaml
    ```
    Configs for different projector architectures are provided in `configs/`:

    | Config | Projector | ImageNet-1k Top-1 |
    |---|---|---|
    | `cc3m_llava_config.yaml` | MLP (4-layer, BatchNorm) | 49.95% |
    | `cc3m_qformer.yaml` | QFormer (2L, 32 queries) | 48.72% |
    | `cc3m_nv_former.yaml` | NV-Former (1L, 32 queries) | 42.80% |
    | `cc3m_mlp_v2.yaml` | MLPv2 (LayerNorm + GELU) | 41.75% |

3. **Zero-Shot ImageNet-1k Evaluation**:
    ```bash
    python eval_zero_shot_imagenet.py \
        --checkpoint logs/<experiment>/version_X/checkpoints/best_model.ckpt \
        --config configs/cc3m_llava_config.yaml
    ```
    Class prototype embeddings are cached automatically (as `<checkpoint>_protos_openai_80tmpl.pt`),
    making repeated evaluations fast (~50 sec image eval after the initial ~1:30 min Llama encoding).

    To benchmark all checkpoints across 2 GPUs in parallel:
    ```bash
    ./run_all_evals.sh
    ```

4. **Inference**:
    The `ShareLock` class implements `encode_text` and `encode_image` for downstream VLM tasks:
    ```python
    from sharelock.models.model import ShareLock
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/cc3m_llava_config.yaml")
    model = ShareLock.load_from_checkpoint("checkpoints/ShareLock-CC3M.ckpt", config=config)
    model.eval()

    image_emb = model.encode_image(image)   # [D]
    text_emb  = model.encode_text("a photo of a cat")  # [D]
    ```

---

## 📂 Pretrained Model Checkpoints

We provide pretrained checkpoints for ShareLock on Hugging Face for easy integration and experimentation:

- **ShareLock (CC3M-trained):** [![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-ShareLock_CC3M-orange)](https://huggingface.co/FunAILab/ShareLock/blob/main/ShareLock-CC3M.ckpt)
- **ShareLock (CC12M-trained):** [![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-ShareLock_CC12M-orange)](https://huggingface.co/FunAILab/ShareLock/blob/main/ShareLock-CC12M.ckpt)

You can load these models directly using the `ShareLock` class:
```python
from sharelock.models.model import ShareLock

model = ShareLock.load_from_checkpoint("path/to/checkpoint.ckpt", config=config)

```
Alternatively, the `--checkpoint` flag can be passed to the `train.py` file. 

---

## 📊 Results

### Paper Results

Our reported results were obtained via the [CLIP-Benchmark](https://github.com/LAION-AI/CLIP_benchmark/tree/main) codebase. A subset of classification results is presented in the following table:

**Zero-shot classification on ImageNet variants:**

| Model         | Dataset        | IN-1k | IN-R  | IN-A  |
|---------------|----------------|-------|-------|-------|
| [CLIP](https://arxiv.org/abs/2103.00020)          | CC3M           | 16.0% | 17.6% | 3.6%  |
| [LiT](https://arxiv.org/abs/2111.07991)           | CC3M           | 46.8% | 72.8% | 59.4% |
| [**ShareLock**](https://arxiv.org/abs/2410.07173) | CC3M           | **54.5%** | **74.7%** | **65.9%** |
| [CLIP](https://arxiv.org/abs/2103.00020)          | CC12M           | 41.6% | 52.6% | 3.6%  |
| [LiT](https://arxiv.org/abs/2111.07991)           | CC12M           | 59.9% | **79.9%** | 68.2% |
| [**ShareLock**](https://arxiv.org/abs/2410.07173) | CC12M           | **62.0%** | 78.5% | **70.1%** |

For a comprehensive and detailed evaluation of ShareLock across various vision-language-modelling tasks, see [our paper](https://arxiv.org/pdf/2410.07173).

### Reproduced Results (Zero-Shot ImageNet-1k)

Evaluated using 80-template CLIP ensemble + OpenAI curated class names on 50K validation images.
All runs on a single NVIDIA RTX PRO 6000 Blackwell (97 GB), batch size 512.

| Checkpoint | Projector | Top-1 | Top-5 |
|---|---|---|---|
| `ShareLock-CC3M.ckpt` (published) | MLP 4-layer + BatchNorm | **51.47%** | **81.29%** |
| `cc3m_llava` (best trained run) | MLP 4-layer + BatchNorm | 49.95% | 79.51% |
| `cc3m_qformer` | QFormer (2L, 32 queries) | 48.72% | 78.41% |
| `cc3m_nv_former` | NV-Former (1L, 32 queries) | 42.80% | 73.91% |
| `cc3m_mlp_v2` | MLPv2 (LayerNorm + GELU) | 41.75% | 73.36% |

> The ~3% gap to the paper's 54.5% is attributed to checkpoint quality differences; the evaluation
> protocol (templates, class names, transforms) matches CLIP-Benchmark exactly.

See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) for full details including timing and training summaries.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{ruthardt2024sharelock,
  title={Better Language Models Exhibit Higher Visual Alignment},
  author={Jona Ruthardt and Gertjan J. Burghouts and Serge Belongie and Yuki M. Asano},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026}
}
```

## 📧 Contact

For any questions or collaborations, contact [Jona Ruthardt](mailto:jona@ruthardt.de).

---

📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
