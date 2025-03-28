# ArtFL Offical Source Code

### ArtFL: Exploiting Data Resolution in Federated Learning for Dynamic Runtime Inference via Multi-Scale Training


### News
[[Paper 🤗](http://syjiang.com/wp-content/uploads/2024/06/IPSN24_Arxiv.pdf)] [[Demo Video🤗](https://youtu.be/eeK6yRVEG3U)] 

- 🔥 Upload the training record of CIFAR-10
- 🏆 ArtFL got **Best Paper Award** at **IPSN 2024**
- 🔥 Release Sample Code


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
```
@inproceedings{jiang2024artfl,
  title={ArtFL: Exploiting data resolution in federated learning for dynamic runtime inference via multi-scale training},
  author={Jiang, Siyang and Shuai, Xian and Xing, Guoliang},
  booktitle={2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
  pages={27--38},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgement
[BalanceFL (IPSN 2022)](https://github.com/sxontheway/BalanceFL)

## License
MIT License.
