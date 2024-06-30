# 2022-10-20 / 4:57PM in CPII dir = 0.1
# 2024-06-28 / 13:01 in CUHK - SY

#################################################################
# Note that if you meet a problem of:                           #
#  - Errno 24: Too many open files. But I am not opening files? #
# input limit -n 64000 in CLI                                   #                                    
#################################################################

### FedAvg: In round 199 => Acc in [(small/Medium/Large), Smooth in Large Acc) =>  [0.283, 0.477, 0.54 ]), 0.5596].
# python3 train_branch.py --cfg=./config/cifar10/fedavg.yaml

### FedProx: In round 199 => Acc in [(small/Medium/Large), Smooth) =>  [0.311, 0.502, 0.579]), 0.5763].
# python3 train_branch.py --cfg=./config/cifar10/fedprox.yaml

### Fedbn: In round 199 => Acc in [(small/Medium/Large), Smooth) =>  [0.275, 0.491, 0.555]), 0.5539].
# python3 train_branch.py --cfg=./config/cifar10/fedbn.yaml

### Feddyn:  In round 199: Acc in [(small/Medium/Large), Smooth) => ([0.289, 0.525, 0.632]), 0.6195]
## Note that Feddyn is not stable, if crash in training, just re-launch the training process
# python3 train_branch.py --cfg=./config/cifar10/feddyn.yaml

### Balancefl: Acc in [(small/Medium/Large), Smooth) => [0.344, 0.534, 0.608]), 0.6106]
# python3 train_branch.py --cfg=./config/cifar10/balancefl.yaml

### centralized:  In round 199 => Acc in [(small/Medium/Large), Smooth) =>  [0.452, 0.764, 0.841], 0.841
# python3 train_branch.py --cfg=./config/cifar10/centralized.yaml

### local:  Acc in [(small/Medium/Large), Smooth) => [0.145, 0.145, 0.149]), 0.1495
# python3 train_branch.py --cfg=./config/cifar10/local.yaml

### ArtFL: In round 199 => Acc in [(small/Medium/Large), Smooth) => [0.509, 0.616, 0.631]), 0.6228]
python3 train_branch.py --cfg=./config/cifar10/artfl.yaml



