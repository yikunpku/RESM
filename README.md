<div align="center">
<h1>🧬 RESM: RNA Evolution-Scale Modeling</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

</div>

## 📣 News
- **[2025/07]** 🎉 We release the dataset and checkpoints on [Zenodo](https://doi.org/10.5281/zenodo.15980875)!
- **[2025/07]** 📊 Initial release of RESM-150M and RESM-650M models with comprehensive documentation.

## ⚡ Overview

RESM (RNA Evolution-Scale Modeling) is a state-of-the-art RNA language model that leverages protein language model knowledge to overcome RNA's inherent challenges. By mapping RNA sequences to pseudo-protein representations and adapting the ESM2 protein language model, RESM provides a robust foundation for deciphering RNA sequence-structure-function relationships.

<div align="center">
<img src="figures/RESM.png" width="800px">
</div>

### Key Features:
- **Pseudo-protein Mapping**: Novel approach to convert RNA's 4-letter alphabet into protein-like representations
- **Knowledge Transfer**: Leverages the powerful representations learned by ESM protein language models
- **Dual-task Excellence**: First RNA model to achieve state-of-the-art performance on both structural and functional prediction tasks
- **Zero-shot Capability**: Outperforms **14** RNA language models in zero-shot evaluation without task-specific training
- **Benchmark Performance**: Demonstrates superior results across **8** downstream tasks, surpassing **60+** models
- **Flexible Architecture**: Available in 150M and 650M parameter versions

## 📥 Download URL

| Resource | Description | Size | Link |
|----------|-------------|------|------|
| **Datasets** | pre-training and downstream datasets | ~6.4GB | [Download](https://doi.org/10.5281/zenodo.15980875) |
| **RESM-150M** |  model checkpoint | ~1.7GB | [Download](https://doi.org/10.5281/zenodo.15980875) |
| **RESM-650M** |  model checkpoint | ~2.5GB | [Download](https://doi.org/10.5281/zenodo.15980875) |


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RESM.git
cd RESM
```
2. Create and activate conda environment:
```bash
# Create conda environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate resm
```

## 📊 Usage

### Feature Extraction

Extract RNA embeddings and attention maps from your RNA sequences:

```bash
# For RESM-150M model (default paths)
python resm_inference.py \
    --base_model RESM_150M \
    --data_path /path/to/your/data \
    --output_dir /path/to/output \
    --device cuda

# For RESM-650M model (default paths)
python resm_inference.py \
    --base_model RESM_650M \
    --data_path /path/to/your/data \
    --output_dir /path/to/output \
    --device cuda

# Use custom checkpoint path
python resm_inference.py \
    --base_model RESM_150M \
    --model_path /path/to/custom/checkpoint.ckpt \
    --data_path /path/to/your/data \
    --output_dir /path/to/output \
    --device cuda
```

### Input Data Format

The model expects RNA sequences in FASTA format or as a text file with RNA IDs. Place your data in the following structure:

```
data/
├── dsdata/
│   ├── msa/          # MSA files (optional, can use single sequences)
│   └── extract_ss_data_alphaid.txt  # List of RNA IDs
```

### Output Format

The model outputs two types of features for each RNA sequence:

1. **Embeddings** (`*_emb.npy`): 
   - RESM-150M: Shape `(L, 640)` where L is sequence length
   - RESM-650M: Shape `(L, 1280)` where L is sequence length

2. **Attention Maps** (`*_atp.npy`): 
   - RESM-150M: Shape `(600, L, L)` (30 layers × 20 heads)
   - RESM-650M: Shape `(660, L, L)` (33 layers × 20 heads)



## 🏗️ Model Architecture

RESM builds upon ESM2 architecture with RNA-specific adaptations:

### RESM-150M (Based on ESM2-150M)
- **Base Model**: `esm2_t30_150M_UR50D`
- **Layers**: 30 transformer layers
- **Embedding Dimension**: 640
- **Attention Heads**: 20
- **Parameters**: ~150M

### RESM-650M (Based on ESM2-650M)
- **Base Model**: `esm2_t33_650M_UR50D`
- **Layers**: 33 transformer layers
- **Embedding Dimension**: 1280
- **Attention Heads**: 20
- **Parameters**: ~650M


## 🔍 Example Use Cases

1. **RNA Secondary Structure Prediction**: Use extracted attention maps for predicting RNA base pairs with state-of-the-art accuracy
2. **RNA Function Classification**: Leverage embeddings for functional annotation of novel RNA sequences
3. **Gene Expression Prediction**: Apply RESM features for mRNA expression level prediction
4. **Ribosome Loading Efficiency**: Predict translation efficiency from mRNA sequences
5. **RNA Similarity Search**: Compare RNA sequences using embedding similarity
6. **Transfer Learning**: Fine-tune on your specific RNA task for enhanced performance

## 📝 Citation

If you use RESM in your research, please cite our paper:

<!-- ```bibtex
@article{zhang2024resm,
  title={RESM: Capturing sequence and structure encoding of RNAs by mapped transfer learning from evolution-scale-modeling protein language model},
  author={Zhang, Y.K. and others},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/xxxx.xxxx}
}
``` -->


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👍 Acknowledgments

- [ESM models](https://github.com/facebookresearch/esm) The codebase we built upon.

  
## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaborations, please contact: yikun.zhang@stu.pku.edu.cn

