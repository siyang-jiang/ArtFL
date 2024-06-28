# ArtFL Offical Source Code

### ArtFL: Exploiting Data Resolution in Federated Learning for Dynamic Runtime Inference via Multi-Scale Training


### News
[[Paper 🤗](http://syjiang.com/wp-content/uploads/2024/06/IPSN24_Arxiv.pdf)] [[Demo Video🤗](https://youtu.be/eeK6yRVEG3U)] 
- 2024-06-25 Release Sample Code🔥
- 2024-05-16 Best Paper Award in IPSN 2024🔥



### Install
1. Clone the repo into a local folder.
```bash
git clone https://github.com/siyang-jiang/ArtFL.git

cd ArtFL
```


```
├── baselines_depracated
│   ├── central_mini.py
│   ├── fed_avg_mini.py
│   ├── fed_branch_backup.py
│   ├── fed_per.py
│   ├── fed.py
│   ├── gen_speech_data.py
│   ├── train_central_bal.py
│   ├── train_central.py
│   ├── train_fedavg.py
│   ├── train_fedprox.py
│   ├── train_FL_mp.py
│   ├── train_local.py
│   ├── train_lth.py
│   └── train_per.py
├── config
│   ├── branch_statefarm.yaml
│   ├── cifar10
│   │   ├── balancefl.yaml
│   │   ├── centralized.yaml
│   │   ├── fedavg.yaml
│   │   ├── fedbn.yaml
│   │   ├── feddyn.yaml
│   │   ├── fedprox.yaml
│   │   └── local.yaml
│   └── default.yaml
├── data
│   ├── aug.py
│   ├── dataloader.py
│   ├── ImbalanceCIFAR.py
│   ├── ImbalanceSpeech.py
│   ├── __init__.py
│   └── __pycache__
│       ├── aug.cpython-39.pyc
│       ├── dataloader.cpython-39.pyc
│       ├── ImbalanceSpeech.cpython-39.pyc
│       └── __init__.cpython-39.pyc
├── dataloader
│   ├── __init__.py
│   ├── LoaderCifar_deprecated.py
│   ├── LoaderCifar.py
│   ├── loader_mini_imagenet.py
│   └── __pycache__
│       ├── __init__.cpython-39.pyc
│       └── LoaderCifar.cpython-39.pyc
├── dataset
│   ├── cifar10.py
│   ├── cifar.py
│   ├── gen_data_depth.py
│   ├── gen_data_statefarm.py
│   ├── mini_imagenet.py
│   ├── __pycache__
│   │   ├── cifar.cpython-39.pyc
│   │   ├── mini_imagenet.cpython-39.pyc
│   │   └── tiny_imagenet.cpython-39.pyc
│   ├── split_data_depth.py
│   └── tiny_imagenet.py
├── environment.yml
├── exp_results_cifar10
│   └── fedavg_0.1_10
│       ├── cfg.yaml
│       ├── log.txt
│       └── tensorboard
│           └── events.out.tfevents.1719550835.SyjiangCUHK.3682881.0
├── fed_branch.py
├── loss
│   ├── BalancedCE.py
│   ├── FocalLoss.py
│   ├── KDLoss.py
│   ├── LwFloss.py
│   ├── __pycache__
│   │   ├── KDLoss.cpython-39.pyc
│   │   └── LwFloss.cpython-39.pyc
│   ├── SoftmaxLoss.py
│   └── WeightedSoftmaxLoss.py
├── models
│   ├── legacy
│   │   ├── DotProductClassifier.py
│   │   ├── ResNet32Feature.py
│   │   ├── ResNetFeature.py
│   │   └── TauNormClassifier.py
│   ├── model.py
│   ├── __pycache__
│   │   ├── model.cpython-39.pyc
│   │   ├── utils.cpython-36.pyc
│   │   └── utils.cpython-39.pyc
│   └── utils.py
├── __pycache__
│   └── fed_branch.cpython-39.pyc
├── README.md
├── requirements.txt
├── run.sh
├── src
│   ├── concat_dataset.py
│   └── partition_dataset.py
├── train_branch.py
└── utils
    ├── __init__.py
    ├── logger.py
    ├── misc.py
    ├── parameters.py
    ├── __pycache__
    │   ├── __init__.cpython-39.pyc
    │   ├── logger.cpython-39.pyc
    │   ├── misc.cpython-39.pyc
    │   ├── sampling.cpython-39.pyc
    │   ├── train_helper.cpython-39.pyc
    │   └── utils.cpython-39.pyc
    ├── sampling.py
    ├── train_helper.py
    └── utils.py
```

2. Install packages.
```
- conda env create -f environment.yml
- conda activate exFL
```

### Quick Start
Before run the bash, make sure you have already prepare the right path to the dataset.
- Change the data path: utils/utils.py change the path and make the data right

```
- bash run.sh
```


