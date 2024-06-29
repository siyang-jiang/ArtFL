# 2022-10-20 / 4:57PM in CPII dir = 0.1
# 2024-06-28 / 13:01 in CUHK - SY

#################################################################
# Note that if you meet a problem of:                           #
#  - Errno 24: Too many open files. But I am not opening files? #
# input limit -n 64000 in CLI                                   #                                    
#################################################################

### FedAvg: In round 199 => Acc in [(small/Medium/Large), Avg) =>  [0.283, 0.477, 0.54 ]), 0.5403].
# python3 train_branch.py --cfg=./config/cifar10/fedavg.yaml

### FedProx: In round 199 => Acc in [(small/Medium/Large), Avg) =>  [0.311, 0.502, 0.579]), 0.5785].
# python3 train_branch.py --cfg=./config/cifar10/fedprox.yaml

### Fedbn: In round 199 => Acc in [(small/Medium/Large), Avg) =>  [0.275, 0.491, 0.555]), 0.5553].
# python3 train_branch.py --cfg=./config/cifar10/fedbn.yaml

### balancefl: 
python3 train_branch.py --cfg=./config/cifar10/balancefl.yaml

### centralized: 
python3 train_branch.py --cfg=./config/cifar10/centralized.yaml

### local: 
python3 train_branch.py --cfg=./config/cifar10/local.yaml

### ArtFL: In round 153 => Acc in [(small/Medium/Large), Avg) => [0.52 , 0.608, 0.622]), 0.6223].
python3 train_branch.py --cfg=./config/cifar10/artfl.yaml

### Feddyn:
python3 train_branch.py --cfg=./config/cifar10/feddyn.yaml