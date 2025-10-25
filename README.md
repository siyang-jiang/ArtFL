# ArtFL: Adaptive Resolution Training for Federated Learning

[![Paper](https://img.shields.io/badge/Paper-IPSN%202024-blue)](http://syjiang.com/wp-content/uploads/2024/06/IPSN24_Arxiv.pdf)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red)](https://youtu.be/eeK6yRVEG3U)
[![Slides](https://img.shields.io/badge/Slides-PDF-orange)](slides/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Best Paper](https://img.shields.io/badge/Award-Best%20Paper%20IPSN%202024-gold.svg)](https://ipsn.acm.org/2024/)

**ArtFL: Exploiting Data Resolution in Federated Learning for Dynamic Runtime Inference via Multi-Scale Training**

## 📰 News

- 🔥 **2024-06**: Uploaded training records for CIFAR-10 experiments
- 🏆 **2024-05**: ArtFL received **Best Paper Award** at **IPSN 2024**
- 🔥 **2024-03**: Released sample code and pre-trained models

## 🎯 Overview

ArtFL is a novel federated learning framework that enables dynamic runtime inference by training models with multiple input resolutions. This approach allows edge devices to adaptively select the appropriate resolution based on their computational constraints while maintaining model accuracy.

### Key Features

- **Multi-Scale Training**: Simultaneously trains models on multiple input resolutions (16×16, 24×24, 32×32)
- **Adaptive Resolution Selection**: Dynamically allocates resolutions based on client capabilities and time constraints
- **Heterogeneous Client Support**: Handles diverse client devices with varying computational resources
- **Multiple FL Algorithms**: Supports FedAvg, FedProx, FedBN, FedDyn, and custom aggregation methods
- **Comprehensive Evaluation**: Includes per-class and per-resolution accuracy tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU training)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siyang-jiang/ArtFL.git
   cd ArtFL
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate exFL
   ```

   Or install packages manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure dataset path:**
   
   Edit `utils/utils.py` to set your dataset path:
   ```python
   # Update the GetDatasetPath function with your local paths
   def GetDatasetPath(config):
       data_root_dict = {
           'CIFAR10': '/path/to/your/cifar10',
           'CIFAR100': '/path/to/your/cifar100',
           # ... add other datasets
       }
       return root_all, data_root_dict
   ```

### Training

Run the default CIFAR-10 experiment:
```bash
bash run.sh
```

Or train with specific configuration:
```bash
python train_branch.py --cfg config/cifar10/artfl.yaml
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=./exp_results_cifar10
```

Open your browser and navigate to `http://localhost:6006`

## 📊 Experiments

### Available Configurations

The `config/cifar10/` directory contains configurations for different FL methods:

- `artfl.yaml` - Our proposed ArtFL method
- `fedavg.yaml` - Federated Averaging baseline
- `fedprox.yaml` - FedProx with proximal term
- `fedbn.yaml` - FedBN with batch normalization
- `feddyn.yaml` - FedDyn with dynamic regularization
- `balancefl.yaml` - BalanceFL for imbalanced data
- `local.yaml` - Local training (no federation)
- `centralized.yaml` - Centralized training baseline

### Running Different Methods

```bash
# ArtFL (our method)
python train_branch.py --cfg config/cifar10/artfl.yaml

# FedAvg baseline
python train_branch.py --cfg config/cifar10/fedavg.yaml

# FedProx
python train_branch.py --cfg config/cifar10/fedprox.yaml
```

### Key Configuration Parameters

Edit YAML files to customize experiments:

```yaml
fl_opt:
  num_clients: 10          # Number of federated clients
  rounds: 600              # Total training rounds
  local_ep: 5              # Local epochs per round
  local_bs: 64             # Local batch size
  frac: 0.5                # Fraction of clients per round
  aggregation: "artfl"     # Aggregation method

hetero_size:
  sizes: [16, 24, 32]      # Input resolutions
  train_hetero: true       # Enable multi-scale training
  level: 0                 # Heterogeneity level (0: high, 2: low)

dataset:
  name: "CIFAR10"          # Dataset name
  dirichlet: 0.1           # Non-IID parameter (lower = more skewed)
  imb_ratio: 1             # Class imbalance ratio
```

## 📁 Project Structure

```
ArtFL/
├── config/                    # Configuration files
│   └── cifar10/              # CIFAR-10 configs
├── data/                      # Data processing
│   ├── dataloader.py         # Main dataloader
│   ├── aug.py                # Data augmentation
│   └── ImbalanceCIFAR.py     # Imbalanced dataset handling
├── dataloader/               # Legacy dataloaders
├── dataset/                  # Dataset implementations
│   ├── cifar10.py
│   ├── tiny_imagenet.py
│   └── ...
├── models/                   # Model architectures
│   ├── model.py             # ResNet variants
│   └── utils.py             # Model utilities
├── loss/                     # Loss functions
│   ├── KDLoss.py           # Knowledge distillation
│   ├── FocalLoss.py        # Focal loss
│   └── ...
├── utils/                    # Utility functions
│   ├── train_helper.py     # Training helpers
│   ├── logger.py           # Logging utilities
│   └── sampling.py         # Client sampling
├── fed_branch.py            # Federated learning core
├── train_branch.py          # Main training script
├── run.sh                   # Quick start script
├── environment.yml          # Conda environment
└── requirements.txt         # Pip requirements
```

## 🔬 Supported Datasets

- **CIFAR-10/100**: Image classification (32×32 images, 10/100 classes)
- **Tiny ImageNet**: Image classification (64×64 images, 200 classes)
- **Mini ImageNet**: Few-shot learning (84×84 images, 100 classes)
- **Speech Commands**: Audio classification (35 classes)
- **IMU**: Human activity recognition (5 activities)
- **State Farm**: Distracted driver detection (10 classes)

## 📈 Results

ArtFL achieves significant improvements over baseline methods:

| Method | CIFAR-10 Accuracy | Computation Cost | Waiting Time |
|--------|-------------------|------------------|--------------|
| FedAvg | 78.2% | 100% | 100% |
| FedProx | 79.1% | 100% | 100% |
| **ArtFL** | **82.4%** | **67%** | **43%** |

See `exp_results_cifar10/` for detailed training logs.

## 🛠️ Extending ArtFL

### Adding a New Dataset

1. Implement dataloader in `data/dataloader.py`:
   ```python
   def YourDataset_FL(root, config):
       # Load and partition data
       return per_client_data, per_client_label, test_data, test_label, ...
   ```

2. Add dataset loading in `train_branch.py`:
   ```python
   elif dataset == "yourdataset":
       (per_client_data, per_client_label, ...) = YourDataset_FL(data_root, config)
   ```

3. Create configuration file in `config/yourdataset/artfl.yaml`

### Adding a New Aggregation Method

1. Implement aggregation in `fed_branch.py`:
   ```python
   def your_aggregation(self, selected_idx):
       # Your aggregation logic
       pass
   ```

2. Update `aggregate_layers` method to support your method

3. Create configuration with `aggregation: "yourmethod"`

## 📝 Citation

If you use ArtFL in your research, please cite our paper:

```bibtex
@inproceedings{jiang2024artfl,
  title={ArtFL: Exploiting data resolution in federated learning for dynamic runtime inference via multi-scale training},
  author={Jiang, Siyang and Shuai, Xian and Xing, Guoliang},
  booktitle={2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
  pages={27--38},
  year={2024},
  organization={IEEE}
}
```

## 🙏 Acknowledgements

This work builds upon:
- [BalanceFL (IPSN 2022)](https://github.com/sxontheway/BalanceFL) - Foundation for federated learning implementation
- PyTorch and torchvision teams for excellent deep learning frameworks

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: syjiang [AT] ie.cuhk.edu.hk

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy Training! 🚀**
