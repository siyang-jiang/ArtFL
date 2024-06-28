# 2022-10-20 / 4:57PM in CPII dir = 0.1
# 2024-06-28 / 13:01 in CUHK - SY

### FedAvg:
# python3 train_branch.py --cfg=./config/cifar10/fedavg.yaml

### FedProx:
# python3 train_branch.py --cfg=./config/cifar10/fedprox.yaml

### Fedbn
# python3 train_branch.py --cfg=./config/cifar10/fedbn.yaml

### Fedbn
# python3 train_branch.py --cfg=./config/cifar10/fedbn.yaml

### Feddyn
# python3 train_branch.py --cfg=./config/cifar10/feddyn.yaml

### balancefl
# python3 train_branch.py --cfg=./config/cifar10/balancefl.yaml

### ArtFL 
# In round 153 => Acc in [(small/Medium/Large), Avg) => [0.52 , 0.608, 0.622]), 0.6223].
python3 train_branch.py --cfg=./config/cifar10/artfl.yaml
