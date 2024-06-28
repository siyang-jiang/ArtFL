from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dataset.cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from torchvision import transforms
from data.dataloader import gen_fl_data

def load_CIFAR(root, cifar_select, train, num_classes, shot_num):
    """
    Load dataset CIFAR into memory. Shot version.
    """
    if num_classes > 10 and cifar_select == "CIFAR10":
        raise RuntimeError

    if train:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "train", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "train", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        else:
            raise RuntimeError
    else:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None)
        else:
            raise RuntimeError

    ###############################################
    ####### load the whole dataset into RAM #######
    ###############################################
    # without transformation
    if cifar_select == "CIFAR10":  # 5000*10+1000*10
        num_per_cls = 5000
        if not train:
            num_per_cls = 1000
            shot_num = 1000
    elif cifar_select == "CIFAR100":  # 500*100+100*100
        num_per_cls = 500
        if not train:
            num_per_cls = 100
            shot_num = 100

    # transformation: data are pre-loaded and do not support augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    all_imgs = []
    all_targets = []
    cnt = 0
    for img, label in zip(dataset.data, dataset.labels):
        cnt += 1
        if train and cnt % num_per_cls >= shot_num:
            continue
        if train:
            all_imgs.append(train_transform(Image.fromarray(img)))
        else:
            all_imgs.append(test_transform(Image.fromarray(img)))
        all_targets.append(label)

    return all_imgs, all_targets


def load_CIFAR_imb(root, cifar_select, train, num_classes, imb_ratio):
    """
    Load CIFAR into memory. Imbalance Version.
    """
    if num_classes > 10 and cifar_select == "CIFAR10":
        raise RuntimeError

    if train:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "train", imbalance_ratio=imb_ratio, root=root, test_imb_ratio=None, reverse=None,
            )
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "train", imbalance_ratio=imb_ratio, root=root, test_imb_ratio=None, reverse=None,
            )
        else:
            raise RuntimeError
    else:
        if cifar_select == "CIFAR100":
            dataset = IMBALANCECIFAR100(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None
            )
        elif cifar_select == "CIFAR10":
            dataset = IMBALANCECIFAR10(
                "test", imbalance_ratio=1, root=root, test_imb_ratio=None, reverse=None
            )
        else:
            raise RuntimeError

    # print("Number of training items per class: ", dataset.get_cls_num_list())

    # transformation: data are pre-loaded and do not support augmentation
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    all_imgs = []
    all_targets = []
    cnt = 0
    for img, label in zip(dataset.data, dataset.labels):
        cnt += 1
        all_imgs.append(img)    # img here is numpy
        # if train:
        #     all_imgs.append(train_transform(Image.fromarray(img)))
        # else:
        #     all_imgs.append(test_transform(Image.fromarray(img)))
        all_targets.append(label)

    return all_imgs, all_targets, dataset.get_cls_num_list()


def CIFAR_FL(root, config):
    """
    Divide CIFAR dataset into small parts for FL.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return:
    ---
    per_client_data, per_client_label: list of lists
    test_data, test_label: both are lists
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]  # 10
    imb_ratio = config["dataset"]["imb_ratio"]  # 0.01, 0.1, 1

    # training
    cifar_select = config["dataset"]["name"]  # CIFAR10

    # An example: 
    # train_data_all: [12406, 3, 32, 32]; train_data_all -> List; train_label_all: 12406;
    # train_num_per_cls: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    train_data_all, train_label_all, train_num_per_cls = load_CIFAR_imb(
        root, cifar_select, train=True, num_classes=num_classes, imb_ratio=imb_ratio
    )

    # test
    test_data, test_label, test_num_per_cls = load_CIFAR_imb(
        root, cifar_select, train=False, num_classes=num_classes, imb_ratio=imb_ratio
    )

    # generate per-client FL data
    idx_per_client, cls_per_client, num_per_cls_per_client \
        = gen_fl_data(train_label_all, train_num_per_cls, config)

    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in idx_per_client[client_i]:
            per_client_data[client_i].append(train_data_all[j])
            per_client_label[client_i].append(train_label_all[j])

    return per_client_data, per_client_label, test_data, test_label, cls_per_client, num_per_cls_per_client, train_num_per_cls


def CIFAR_FL_mixed(root, config):
    """
    Divide CIFAR dataset into small parts for FL to validate the algorithms.
    (iid + many shot) for half of all classes; (non-iid + few shot) for remaining half classes.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return:
    ---
    per_client_data, per_client_label: list of lists
    test_data, test_label: both are lists
    """
    shot_num = config['dataset']['shot']
    few_shot_num = config['dataset']['shot_few']
    assert (shot_num != 0 and few_shot_num != 0)

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]

    # training
    cifar_select = config["dataset"]["name"]
    train_data_all, train_label_all = load_CIFAR(
        root, cifar_select, train=True, num_classes=num_classes, shot_num=shot_num
    )
    # test
    test_data, test_label = load_CIFAR(
        root, cifar_select, train=False, num_classes=num_classes, shot_num=shot_num
    )

    # per-client FL data for the first half (iid + many shot) classes
    half_data_len = int(len(train_label_all) / 2)
    iid_train_data_all, iid_train_label_all = train_data_all[:half_data_len], train_label_all[:half_data_len]
    config["dataset"]["non_iidness"] = 0
    iid_idx_per_client, tao, non_iidness, iid_cls_per_client = gen_fl_data(iid_train_label_all, config)
    print("IID, tao:", tao, "non-iidness:", non_iidness)

    # per-client FL data for the remaining half (non-iid + few shot) classes
    noniid_train_data_all = []
    noniid_train_label_all = []
    cnt = 0
    for img, label in zip(train_data_all[half_data_len:], train_label_all[half_data_len:]):
        cnt += 1
        if cnt % shot_num >= few_shot_num:
            continue
        noniid_train_data_all.append(img)
        noniid_train_label_all.append(label)
    config["dataset"]["non_iidness"] = 1
    noniid_idx_per_client, tao, non_iidness, noniid_cls_per_client = gen_fl_data(noniid_train_label_all, config)
    print("Non-IID, tao:", tao, "non-iidness:", non_iidness)

    # iid + non-iid combination
    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in iid_idx_per_client[client_i]:
            per_client_data[client_i].append(iid_train_data_all[j])
            per_client_label[client_i].append(iid_train_label_all[j])
        for j in noniid_idx_per_client[client_i]:
            per_client_data[client_i].append(noniid_train_data_all[j])
            per_client_label[client_i].append(noniid_train_label_all[j])

    cls_per_client = []
    for iid_cls, noniid_cls in zip(iid_cls_per_client, noniid_cls_per_client):
        cls_per_client.append(np.concatenate((iid_cls, noniid_cls)))

    return per_client_data, per_client_label, test_data, test_label, cls_per_client
