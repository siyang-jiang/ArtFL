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
│   ├── cifar10
│   └── default.yaml
├── data
│   ├── aug.py
│   ├── dataloader.py
│   ├── ImbalanceCIFAR.py
│   ├── ImbalanceSpeech.py
│   ├── __init__.py
│   └── __pycache__
├── dataloader
│   ├── __init__.py
│   ├── LoaderCifar_deprecated.py
│   ├── LoaderCifar.py
│   ├── loader_mini_imagenet.py
│   └── __pycache__
├── dataset
│   ├── cifar10.py
│   ├── cifar.py
│   ├── __pycache__
│   └── tiny_imagenet.py
├── environment.yml
├── fed_branch.py
├── LICENSE
├── loss
│   ├── BalancedCE.py
│   ├── FocalLoss.py
│   ├── KDLoss.py
│   ├── LwFloss.py
│   ├── __pycache__
│   ├── SoftmaxLoss.py
│   └── WeightedSoftmaxLoss.py
├── models
│   ├── legacy
│   ├── model.py
│   ├── __pycache__
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
    ├── sampling.py
    ├── train_helper.py
    └── utils.py
```
