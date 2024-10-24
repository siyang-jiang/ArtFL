# ArtFL Offical Source Code

### ArtFL: Exploiting Data Resolution in Federated Learning for Dynamic Runtime Inference via Multi-Scale Training


### News
[[Paper 🤗](http://syjiang.com/wp-content/uploads/2024/06/IPSN24_Arxiv.pdf)] [[Demo Video🤗](https://youtu.be/eeK6yRVEG3U)] 
- **2024-06-29** 🔥 Upload the training record of CIFAR-10
- **2024-06-25** 🔥 Release Sample Code
- **2024-05-16** 🏆 **Best Paper Award** in IPSN 2024 
- **2024-01-20** 🔥 ArtFL is accepted by **IPSN 2024**



### Install
1. Clone the repo into a local folder.
`
git clone https://github.com/siyang-jiang/ArtFL.git

cd ArtFL
`


2. Install packages.
```
conda env create -f environment.yml
conda activate exFL
```

### Quick Start
Before run the bash, make sure you have already prepare the right path to the dataset.
- Change the data path: utils/utils.py change the path and make the data right

```
bash run.sh
```

See the training log in CIFAR-10
```
tensorboard --log_dir=./exp_results_cifar10
```



### Source Tree
```
├── config
├── data
├── dataloader
│   ├── __init__.py
│   ├── LoaderCifar_deprecated.py
│   ├── LoaderCifar.py
│   └── loader_mini_imagenet.py
├── dataset
│   ├── cifar10.py
│   ├── cifar.py
│   └── tiny_imagenet.py
├── environment.yml
├── fed_branch.py
├── LICENSE
├── loss
├── models
├── README.md
├── requirements.txt
├── run.sh
├── src
├── train_branch.py
└── utils
```

## Citation

## Acknowledgement
[BalanceFL (IPSN 2022)](https://github.com/sxontheway/BalanceFL)

## License
MIT License.
