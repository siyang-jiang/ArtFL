from distutils.command.config import config
import sys
from urllib.parse import ParseResultBytes
sys.path.append('../dataset')
# from dataset.mini_imagenet import miniImagenet
from dataset.tiny_imagenet import ImageFolder_custom


from collections import Counter
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision import transforms
import os, random
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from .aug import RandAugment

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size).squeeze(0)
    return image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def non_iidness_cal(labels, idx_per_client, img_per_client):
    """
    Argu:
        labels: list with length of n, where n is the dataset size.
        idx_per_client: list. idx_per_client[i] is the img idx in the dataset for client i
        img_per_client: list. Number of images per client.
    Return:
        - non_iidness
    """
    client_num = len(idx_per_client)
    class_num = max(labels)+1
    label_per_client_count = np.zeros((client_num, class_num))

    # generate per client label counting
    labels = np.array(labels)
    for i in range(client_num):
        count = np.bincount(labels[idx_per_client[i]])
        count_pad = np.pad(count, (0, class_num-len(count)), 'constant', constant_values=(0,0))
        label_per_client_count[i] += count_pad

    # obtain non_iidness 
    summation = 0
    label_per_client_count /= np.array([img_per_client]).T  # broadcast
    for i, client_i in enumerate(label_per_client_count):
        for client_j in label_per_client_count[i:]:
            summation += np.linalg.norm(client_i-client_j, ord=1)
    
    non_iidness = summation/(client_num*(client_num-1))

    return non_iidness


def tao_sampling(img_per_client, tao, num_per_cls):
    """
    Do non-iid or iid sampling, according to "tao". 
    We will sample number of "tao" images for every client in turn. 
    --- 
    Argu:
        - img_per_client: list. Number of images per client.
        - tao: number of sampled image for each client in each round. 
        We use tao to control the non-iidness. When tao==1, nearly iid; 
        when tao is large, it becomes non-iid. 
        "tao <= min(img_per_client/2)" to let each client has at least 2 classes
        - num_per_cls: number of samples per class
    Return:
        - idx_per_client: list. idx_per_client[i] is the img idx in the dataset for client i
    """
    # prepare parameters
    total_img_num = sum(img_per_client)
    client_num = len(img_per_client)
    idx_per_client = [[] for i in range(client_num)]
    # assert tao <= min(img_per_client)/2 
    
    available_per_client = img_per_client
    tao_count = 0
    client_k = 0
    idx = 0
    client_count = 0
    client_order = [*range(client_num)]

    num_per_cls_cum_sum, tmp = [], 0
    for i in range(len(num_per_cls)):
        num_per_cls_cum_sum.append(tmp)
        tmp += num_per_cls[i]

    # assign every samples to a client
    while idx < total_img_num:      

        """
        Original Code：Only Code 3
        Now: uncomment Code 3 , add Code 1 Code2
        """
        # if change to another class, reset tao, <<Code 1>>
        if total_img_num-idx in num_per_cls_cum_sum:    
            tao_count = 0

        client_k = client_order[client_count]
        if available_per_client[client_k] > 0 and tao_count < tao: 
            idx_per_client[client_k].append(total_img_num-idx-1)    # reverse the head and tail
            tao_count += 1
            idx += 1
            available_per_client[client_k] -= 1
        
        # the client is already full, or tao samples are already assigned,  
        # change the client and reset tao
        else:
            client_count = client_count + 1 
            # shuffle the order of clients if a round is finished
            if client_count >= client_num:
                random.shuffle(client_order)
                client_count = 0
                tao_count = 0   #  <<Code 2>>
            # tao_count = 0  #  <<Code 3>>
        continue

    return idx_per_client


def diri_sampling(num_per_client, train_label_all, num_per_cls, diri):
    """
    This sampling scheme CANNOT guarantee that the generated "num_per_client_done" exactly equals the "num_per_client"
    Return: idx_per_client, num_per_client_done
    """
    num_cls = len(num_per_cls)
    np.random.seed(1) 
    min_size, min_require_size = 0, 1000   # min sample amount per client
    client_num = len(num_per_client)

    # ensure every client has at least min_require_size samples
    while min_size < min_require_size:  

        idx_per_client = [[] for _ in range(client_num)]
        num_per_client_done = [0 for _ in range(client_num)]

        for cls_i in range(num_cls):  
            
            idx_cls_i = np.where(np.array(train_label_all) == cls_i)[0]
            np.random.shuffle(idx_cls_i)

            # for every class, divide samples to every client
            proportions = np.random.dirichlet(np.repeat(diri, client_num))

            # balance the sample number among clients
            proportions = np.array([p * (num_done < num) for p, num_done, num in zip(proportions, num_per_client_done, num_per_client)])
            proportions = proportions / proportions.sum()
            
            # split the idx_cls_i based on the proportions
            amount_cum = (np.cumsum(proportions) * len(idx_cls_i)).astype(int)[:-1]
            idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(idx_cls_i, amount_cum))]  
            num_per_client_done = [len(idx_j) for idx_j in idx_per_client]

        min_size = min(num_per_client_done)
        return idx_per_client, num_per_client_done


def gen_fl_data_real(train_label_all, num_per_cls, config):
    pass

def gen_fl_data(train_label_all, num_per_cls, config):
    """
    Generate distributed data for FL training.
    ---
    Argu:
        - train_label_all: object of a class inheriting from torch.utils.data.Dataset 
            Or a list pre-stored in the RAM.
        - num_per_cls: sample number per class
        - config: configuration dictionary
    Return:
        - idx_per_client: list. The i^th item is the img idx of the training set for client i
        - tao: int
        - non_iidness: the calculated non_iidness
    """      
    # generate img_per_client
    client_num = config["fl_opt"]["num_clients"]
    img_per_client_dist = config["dataset"]["img_per_client_dist"]
    total_img_num = len(train_label_all)
    if img_per_client_dist == "uniform":
        num_per_client = np.full(client_num, total_img_num//client_num)
        num_per_client[:total_img_num % client_num] += 1
    else:    # use other num_per_client distributions: normal, LT, reverse LT
        pass

    """ Tao-sampling, deprecated
    # When iid, tao=1. When non_iid, tao=max(num_per_client)
    non_iidness_degree = config["dataset"]["non_iidness"]
    tao_max = min(num_per_client)#//2
    tao = round(1 + non_iidness_degree*(tao_max-1))
    tao = int(config["dataset"]["tao_ratio"] * num_per_cls[-1])
    idx_per_client = tao_sampling(num_per_client.copy(), tao, num_per_cls)
    print("tao:", tao, "non-iidness:", non_iidness)
    """


    """ dirichlet-sampling, deprecated
    # Just copy from the CIFAR-10
    """
    diri = config["dataset"]["dirichlet"]
    idx_per_client, num_per_client = diri_sampling(num_per_client, train_label_all, num_per_cls, diri)

    # calculate the real non_iidness on training set
    non_iidness = non_iidness_cal(train_label_all, idx_per_client, num_per_client)
    print("dirichlet:", diri, "non-iidness:", non_iidness)
    print("FL number per client: ", num_per_client, '\n')
    
    # classes per client
    cls_per_client = []
    num_per_cls_per_client = []

    for idxs in tqdm(idx_per_client):
        cls, tmp = np.unique(np.array(train_label_all)[idxs], return_counts=True)
        num_per_cls = np.zeros(config["dataset"]["num_classes"], dtype=np.int)
        np.put_along_axis(num_per_cls, cls, tmp, axis=0)
        cls_per_client.append(cls)
        num_per_cls_per_client.append(num_per_cls)

    return idx_per_client, cls_per_client, num_per_cls_per_client
    
def all_index_user_in_users(target, list_target):
    result = []
    for idx, value in enumerate(list_target):
        if target == value:
            result.append(idx)
    return result


def gen_fl_data_new(train_label_all, config, users = None):
    """
    Generate distributed data for FL training.
    ---
    Argu:
        - train_label_all: object of a class inheriting from torch.utils.data.Dataset 
            Or a list pre-stored in the RAM.
        - num_per_cls: sample number per class
        - config: configuration dictionary
    Return:
        - idx_per_client: list. The i^th item is the img idx of the training set for client i
        - tao: int
        - non_iidness: the calculated non_iidness
    """      
    # generate img_per_client from the users

    client_num = config["fl_opt"]["num_clients"]
    client_num_real = len(np.unique(users)) # 26个
    print(client_num, client_num_real)
    print("data preprocessing in federated data")

    idx_per_client = [] # 这个client上面的data的索引, Spliting the data according to user ID, 按照users来分
    for user in np.unique(users)[:client_num]: # 仅仅用前面n个client data；
        tmp = all_index_user_in_users(user, users) # find user id in users => 说白了就是找到idx，索引
        idx_per_client.append(tmp) 
    
    cls_per_client = [] # classes per client
    num_per_cls_per_client = []

    for idxs in tqdm(idx_per_client):
        # print(np.array(train_label_all)[idxs])
        cls, tmp = np.unique(np.array(train_label_all)[idxs], return_counts=True)
        num_per_cls = np.zeros(config["dataset"]["num_classes"], dtype=np.int)
        np.put_along_axis(num_per_cls, cls, tmp, axis=0)
        cls_per_client.append(cls)
        num_per_cls_per_client.append(num_per_cls)

    return idx_per_client, cls_per_client, num_per_cls_per_client

def size_of_client_i(config, client_id):
    """
    Return the size for client_i
    """
    size_list = config["hetero_size"]["sizes"]
    level = config["hetero_size"]["level"]
    number_clients = config["fl_opt"]["num_clients"]
    size_idx_list, number_per_size = size_sampling(number_clients, level)
    return size_list[size_idx_list[client_id]]


def size_sampling(number_clients, level):
    """
    Sampling the resolution
    CIFAR-10: (0: Hard, 1: Medium, 2: Easy)
    ---
    Return size_idx_list
    """
    if level == 0:  # Hard, e.g. (4,3,3)
        number_per_size = [number_clients-2*(number_clients//3), number_clients//3, number_clients//3]
    if level == 1: # Medium, e.g. (0,5,5)
        number_per_size = [0, number_clients-number_clients//2, number_clients//2]
    if level == 2:  # Easy, e.g. (0,0,10)
        number_per_size = [0, 0, number_clients]
    
    size_idx_list = [0]*number_per_size[0] + [1]*number_per_size[1] + [2]*number_per_size[2]
    return np.array(size_idx_list), np.array(number_per_size)


def aug_sampling(perturb_prob, augs, client_id):
    """
    Sampling the perturbation
    ---
    Args:
    - perturb_prob: probability 0, 0.5, 1
    - augs: an object of the class: RandAugment
    - client_id: int
    Return:
    - list: [] or [idx_aug_1, prob_to_use_fix_aug] or [idx_aug_1, idx_aug_2, prob_to_use_fix_aug]
    """
    aug_space_num = augs.aug_space_len()
    aug_num = augs.num_ops
    if aug_num == 0 or client_id is None:   # None means the server
        transform_augs = []
    if aug_num == 1:
        transform_augs = [client_id % aug_space_num, perturb_prob]
    if aug_num == 2:
        rand = random.randint(0, aug_space_num-1)
        transform_augs = [client_id // aug_space_num, rand, perturb_prob]
        # transform_augs = [client_id // aug_space_num, client_id % aug_space_num, perturb_prob]

    return transform_augs


class local_client_dataset(Dataset):

    def __init__(self, 
                 per_client_data,
                 per_client_label,
                 config,
                 phase,
                 client_id = None,  # will be None for server
                 eval_size_id = 2,   # 2 means the original size
                 size_align = False,
                 size_prob = None,
                 changed_size = None
                 ):      
        
        self.data = per_client_data
        self.label = per_client_label
        self.phase = phase
        self.client_id = client_id  
        self.eval_size_id = eval_size_id
        self.config = config
        self.multiscale = config["fl_opt"]["multi_scale"]
        
        # Others
        self.dataset_name = config["dataset"]["name"]
        self.train_hetero = config["hetero_size"]["train_hetero"]
        self.size_level = config["hetero_size"]["level"]
        self.perturb_prob = config["hetero_pert"]["perturb_prob"]
        self.perturb_num = config["hetero_pert"]["perturb_num"]
        self.sizes = config["hetero_size"]["sizes"]
        self.batch_count = 0
        self.batch_per_epoch = len(self.label)
        self.size_align = size_align
        if client_id is not None:
            self.size_prob = size_prob[client_id]
        self.changed_size = changed_size
        assert self.phase in ["train", "val", "test"]
        assert self.size_level in [0, 1, 2] 
        assert self.perturb_num in [0, 1, 2]
        assert self.perturb_prob in [0, 0.5, 1]

        # get transform
        self.update_transform()

    def update_transform(self):
        """
        Generate Transforms. Order: pertu -> resize -> ToTensor -> normalize
        """
        assert self.eval_size_id in [0,1,2]

        if self.dataset_name in ["CIFAR10", "CIFAR100", "tiny", "Speech", "imu", "statefarm", 'depth']:
            # perturbation
            transform_augs = []
            if self.phase == "train" and self.perturb_num > 0:
                augs = RandAugment(num_ops=self.perturb_num)
                augs.augs_config = aug_sampling(self.perturb_prob, augs, self.client_id)
                transform_augs = [augs]               
                
            # transform resolution
            resize_list = self.sizes
            transform_size = []
            
            if self.phase == "train" and self.config['motivation_flag']:
                transform_size = [transforms.Resize(self.changed_size)]  
                
            if self.phase == "train" and self.train_hetero:
                # we move this function into the collect_fn
                pass 
                # size = size_of_client_i(self.config, self.client_id) # Get train size for clients
                # transform_size = [transforms.Resize(size)] 
                # transform_size = [transforms.Resize(32)]      
            if self.phase in ["val", "test"] and self.eval_size_id != 2:  # 2 means the original size
                # Eval function still works
                size = resize_list[self.eval_size_id]         
                transform_size.append(transforms.Resize(size))
                # transform_size = [transforms.Resize(32)]  
            
            # other transforms
            if self.dataset_name in ['Speech', 'depth']:
                transform_others = [
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])
                ]
            else:
                transform_others = [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            transform_list = transform_size + transform_others
            # transform_list = transform_augs + transform_size + transform_others
            self.transform = transforms.Compose(transform_list)
            # print(self.transform)
            # raise ValueError("1")
            
        elif self.dataset_name == "CUB":
            self.train_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665])
                ])
            self.val_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665])
                ])
        elif self.dataset_name in ["mini"]:
            self.train_transform = transforms.Compose([
                Image.fromarray,
                transforms.Resize(92),
                transforms.RandomCrop(84),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.22)),   
            ])
            self.val_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.22)),   
                ])     
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):      
        data = self.data[index]  # (h,w,c) 


        if self.dataset_name == "Speech" :
            # if multiscale is true, force the index to [0, 43960]
            if self.multiscale and self.phase == 'train' and self.config['use_resize_data']:
                self.max_length = int(len(self.label) / 3 )
                index %= self.max_length
                data = self.data[index]
        
            data = data.squeeze().cpu().numpy()
            
    
        if self.dataset_name in ["statefarm", "depth"] :
            data = np.transpose(data, (1,2,0))
            if data.shape[2] == 1:
                data = np.squeeze(data, axis=2)
            data = data * 255
            data = data.astype(np.uint8)

        assert type(data) is np.ndarray
        if self.config['motivation_flag']:# This line is only for motivation study 
            self.train_transform = self.update_transform()
        transformed_data = self.transform(Image.fromarray(data))

        if self.size_align:
            return data, transformed_data, self.label[index], index
        else:
            return transformed_data, self.label[index], index
        
    def collate_fn(self, batch):
        imgs, targets, indexs = list(zip(*batch)) # Check is the index or not
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % self.config["fl_opt"]["multi_scale_batch"] == 0:
            
            # sample with probability
            tmp = np.array(range(self.sizes[0], self.sizes[-1] + 1, (self.sizes[-1] - self.sizes[0])//2))   
            self.img_size = np.random.choice(tmp, size=1, replace=True, p=self.size_prob)[0]
             
        if self.img_size != self.sizes[self.eval_size_id]:

            if self.dataset_name == 'Speech' and self.phase == 'train' and self.config['use_resize_data']:

                transformed_imgs = []

                for img, _, index in zip(imgs, targets, indexs):
                    if self.img_size == 32:
                        transformed_imgs.append(img)
                        raise ValueError('U did not come in')
                    elif self.img_size == 24:

                        index += self.max_length
                        img = self.data[index]
                        img = img.squeeze().cpu().numpy()
                        img = self.transform(Image.fromarray(img))
                        transformed_imgs.append(img)
                    
                    elif self.img_size == 16:

                        tmp = self.max_length
                        tmp *= 2
                        index += tmp
                        img = self.data[index]
                        img = img.squeeze().cpu().numpy()
                        img = self.transform(Image.fromarray(img))
                        transformed_imgs.append(img)
                    
                    else:
                        raise ValueError("Bad Collate_fn")
                    
                    # img_cp = img.squeeze().cpu().numpy()  # [1, 32, 32] [No B]
                    # H, W = img_cp.shape
                    # if self.img_size != W or self.img_size != H:
                    #     raise ValueError("Not Matching")

                imgs = torch.stack(transformed_imgs)

            else:
                imgs = torch.stack([resize(img, self.img_size) for img in imgs])
               
        else:
            imgs = torch.stack(imgs)
        
        self.batch_count += 1
        targets = torch.tensor(targets)
        indexs = torch.tensor(indexs)
        
        return imgs, targets, indexs

    def get_balanced_sampler(self):
        labels = np.array(self.label, dtype="object")     # e.g., size (n, ), n is the number of client
        unique_labels = list(np.unique(labels))

        transformed_labels = torch.tensor([unique_labels.index(i) for i in labels])         # e.g., size (n, )
        class_sample_count = np.array([len(np.where(labels==t)[0]) for t in unique_labels]) # e.g., size (6, ), 6 classes

        weight = 1. / class_sample_count    # make every class to have balanced chance to be chosen
        samples_weight = torch.tensor([weight[t] for t in transformed_labels])
        self.class_sample_count = class_sample_count
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler


class test_dataset(Dataset):
    def __init__(self, per_client_data, per_client_label, config):
        self.data = per_client_data
        self.label = per_client_label
        self.dataset = config["dataset"]["name"]
        if self.dataset == "CUB":
            self.val_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4856, 0.4994, 0.4324], std=[0.2321, 0.2277, 0.2665])
                ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        
        data = self.data[index]
        if self.dataset == "CUB":
            return self.val_transform(data), self.label[index], index
        else:
            return data, self.label[index], index

"""
Other datasets: miniImageNet, CUB
"""
def load_mini_imagenet_imb(path_image, path_label, dataset_name, num_classes, imb_ratio = None):
    """
    Load mini_imagenet into memory. Imbalance version.
    Input: root is a dictionary for all dataset.
    Return:
    all_imgs: [Number, C, H, W] 
    all_targets: [Number, Label]
    dataset.get_cls_num_list(): The number of images in each class 
    """
    if num_classes > 100 and dataset_name == "mini":
        raise RuntimeError

    dataset = miniImagenet(path_image, path_label)   # miniImageNet loader
    cls_number, image_number_per_cls = dataset.get_cls_num()

    ###############################################
    ####### load the whole dataset into RAM #######
    ###############################################

    # transformation: data are pre-loaded and do not support augmentation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    all_imgs = []
    all_targets = []
    for img, label in zip(dataset.data_set, dataset.label_set):     # img: numpy array
        # img = transform(Image.fromarray(img))
        all_imgs.append(img)    
        all_targets.append(label.detach().numpy())

    cls_number_list = [image_number_per_cls for i in range(cls_number)]
    return all_imgs, all_targets, cls_number_list


def mini_ImageNet_FL(root, config):
    """
    Divide mini_ImageNet dataset into small ones for FL.  
    Argu:
    - mode: base, novel or all classes
    - shot_num: the number of n (n-shot)
    ---
    Return: 
    - per_client_data, per_client_label: list of lists  
    - test_data, test_label: both are lists  
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"] # 10 
    imb_ratio = config["dataset"]["imb_ratio"] # 0.01
    dataset_name = config["dataset"]["name"] 

    # training
    train_data_all, train_label_all, train_num_per_cls = load_mini_imagenet_imb(   
        root["train_image"], root["train_label"], dataset_name, num_classes=num_classes
    ) 

    print("########### Train Mode ################" )
    print("#Train: Number of items per class: ", train_num_per_cls )

    # val
    val_data_all, val_label_all, val_num_per_cls = load_mini_imagenet_imb(
        root["val_image"], root["val_label"], dataset_name, num_classes=num_classes
    )

    print()
    print("########### Val Mode ################" )
    print("#Val: Number of items per class: ", val_num_per_cls )

    # test
    test_data_all, test_label_all, test_num_per_cls = load_mini_imagenet_imb(   
        root["test_image"], root["test_image"], dataset_name, num_classes=num_classes
    ) 

    print()
    print("########### Test Mode ################" )
    print("#Test: Number of items per class: ", test_num_per_cls )

    idx_per_client, cls_per_client, num_per_cls_per_client \
        = gen_fl_data(train_label_all, train_num_per_cls, config)
        
    #  IF split, just split;

    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in idx_per_client[client_i]:
            per_client_data[client_i].append(train_data_all[j]) 
            per_client_label[client_i].append(train_label_all[j]) 

    return  per_client_data, per_client_label, \
            val_data_all, val_label_all, \
            test_data_all, test_label_all, \
            cls_per_client, num_per_cls_per_client, train_num_per_cls


def load_tiny_imagenet_imb(path, dataset_name, num_classes, imb_ratio = None, dataidxs = None):
    """
    Load mini_imagenet into memory. Imbalance version.
    Input: root is a dictionary for all dataset.
    Return:
    all_imgs: [Number, C, H, W] 
    all_targets: [Number, Label]
    dataset.get_cls_num_list(): The number of images in each class 
    """
    # collect all dataset
    if num_classes > 200 and dataset_name == "tiny":
        raise RuntimeError

    dl_obj = ImageFolder_custom
    transform_train = transforms.Compose([
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ds = dl_obj(path, dataidxs=dataidxs, transform=transform_train)
    X , y = ds.datas, ds.targets
    
    
    cls_number_list = [len(X)/ 200 for i in range(200)] # Fixed for tiny-miniImagenet

    return X, y, cls_number_list

def tiny_miniImageNet_FL(root, config):
    """
    Divide mini_ImageNet dataset into small ones for FL.  
    Argu:
    - mode: base, novel or all classes
    - shot_num: the number of n (n-shot)
    ---
    Return: 
    - per_client_data, per_client_label: list of lists  
    - test_data, test_label: both are lists  
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"] # 10 
    imb_ratio = config["dataset"]["imb_ratio"] # 0.01
    dataset_name = config["dataset"]["name"] 

    # data_path = root[dataset_name]
    # print(root)

    # training
    root_train = root+'/train/'
    root_val = root+'/val/'
    # root_test = root+'/test/'

    train_data_all, train_label_all, train_num_per_cls = load_tiny_imagenet_imb(   
        root_train, dataset_name, num_classes=num_classes
    ) 

    print("########### Train Mode ################" )
    print("#Train: Number of items per class: ", train_num_per_cls )

    # val
    val_data_all, val_label_all, val_num_per_cls = load_tiny_imagenet_imb(
        root_val, dataset_name, num_classes=num_classes
    )

    print()
    print("########### Val Mode ################" )
    print("#Val: Number of items per class: ", val_num_per_cls )

    # Val is the test 
    test_data_all, test_label_all, test_num_per_cls = load_tiny_imagenet_imb(   
        root_val,  dataset_name, num_classes=num_classes
    ) 

    print()
    print("########### Test Mode ################" )
    print("#Test: Number of items per class: ", test_num_per_cls )

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

    return  per_client_data, per_client_label, \
            val_data_all, val_label_all, \
            test_data_all, test_label_all, \
            cls_per_client, num_per_cls_per_client, train_num_per_cls

def load_CUB(root, train, num_classes, novel_only, shot_num, aug):
    """
    load dataset CUB into memory
    """
    loader = pil_loader

    img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
    img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
    train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])

    data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
    data = data[data['train_flag'] == train]    # trianing mode or evaluation mode
    data['label'] = data['label'] - 1
    img_folder = os.path.join(root, "images")

    # split dataset for base classes and novel classes.
    data = data[data['label'] < num_classes]
    base_data = data[data['label'] < 100]
    novel_data = data[data['label'] >= 100]     

    # sampling n shot from both base and novel classes
    base_data = base_data.groupby('label', group_keys=False).\
        apply(lambda x: x.iloc[:shot_num])  # n*100 rows, 6 coloumns; n is the shot number
    novel_data = novel_data.groupby('label', group_keys=False).\
        apply(lambda x: x.iloc[:shot_num])  # n*100 rows, 6 coloumns; n is the shot number

    # whether only return data of novel classes
    if novel_only:
        data = novel_data
    else:
        data = pd.concat([base_data, novel_data])

    # repeat 5 times for data augmentation
    if aug:
        tmp_data = pd.DataFrame()
        for i in range(5):
            tmp_data = pd.concat([tmp_data, data])
        data = tmp_data
    imgs = data.reset_index(drop=True)  # https://blog.csdn.net/lujiandong1/article/details/52929090
    if len(imgs) == 0:
        raise(RuntimeError("no csv file"))

    ###############################################
    ####### load the whole dataset into RAM #######
    ###############################################
    all_imgs = []
    all_targets = []
    for index in range(len(imgs)):
        item = imgs.iloc[int(index)]
        file_path = item['path']
        all_imgs.append(loader(os.path.join(img_folder, file_path)))
        all_targets.append(item['label'])

    return all_imgs, all_targets


def CUB_FL(root, mode, config, aug=False):
    """
    Divide CUB dataset into small ones for FL.  
    mode: base, novel or all classes
    Return: 
    ---
    per_client_data, per_client_label: list of lists  
    test_data, test_label: both are lists  
    """
    shot_num = config['dataset']['shot']
    assert shot_num != 0

    # mode selection
    if mode == "base":
        num_classes = config["networks"]["classifier"]["params"]["num_classes"]
        novel_only = False
    elif mode == "novel":
        num_classes = 200
        novel_only = True
    elif mode == "all":
        num_classes = 200
        novel_only = False
    else:
        raise NotImplementedError

    # training
    train_data_all, train_label_all = load_CUB(
        root, train=True, num_classes=num_classes, 
        novel_only=novel_only, shot_num=shot_num, aug=aug
        )
    # test
    test_data, test_label = load_CUB(
        root, train=False, num_classes=num_classes, 
        novel_only=novel_only, shot_num=100, aug=aug
        )
   
    # generate per-client FL data
    idx_per_client, cls_per_client = gen_fl_data(train_label_all, config)

    client_num = config["fl_opt"]["num_clients"]
    per_client_data = [[] for i in range(client_num)]
    per_client_label = [[] for i in range(client_num)]
    for client_i in range(client_num):
        for j in idx_per_client[client_i]:
            per_client_data[client_i].append(train_data_all[j]) 
            per_client_label[client_i].append(train_label_all[j]) 

    return per_client_data, per_client_label, test_data, test_label, cls_per_client



############## Speech ############
def Speech_FL(root, config):
    """
    Divide Speech-Command dataset into small ones for FL.  
    shot_num: the number of n (n-shot)
    Return: 
    ---
    per_client_data, per_client_label: list of lists  
    test_data, test_label: both are lists  
    """    
    num_classes = config["networks"]["classifier"]["params"]["num_classes"]
    imb_ratio = config["dataset"]["imb_ratio"]
    
    # only invoke 'ImbalanceSpeech' for one time during the dataset generation, as it will cause multi-process problems
    data_file_path = f"./data/training_data.pt"
    if os.path.exists(data_file_path) is False:
        from data.ImbalanceSpeech import ImbalanceSpeech
        train_dataset = ImbalanceSpeech("training", imbalance_ratio=imb_ratio, root=root, reverse=None) # imb_ratio only for training set
        test_dataset = ImbalanceSpeech("testing", imbalance_ratio=None, root=root, reverse=None)
        val_dataset = ImbalanceSpeech("validation", imbalance_ratio=None, root=root, reverse=None)
    else :

        # train_dataset = ImbalanceSpeech_clean("training", imbalance_ratio=imb_ratio, root=root, reverse=None)   # imb_ratio only for training set
        # val_dataset = ImbalanceSpeech_clean("validation", imbalance_ratio=None, root=root, reverse=None)
        # test_dataset = ImbalanceSpeech_clean("testing", imbalance_ratio=None, root=root, reverse=None)

        # training
        train_data_all, train_label_all, train_num_per_cls = \
            ImbalanceSpeech_clean("training", imb_ratio, root, config=config)

        # validation
        val_data, val_label, val_num_per_cls = \
            ImbalanceSpeech_clean("validation", None, root, config=config)

        # testing
        test_data, test_label, test_num_per_cls = \
            ImbalanceSpeech_clean("testing", None, root, config=config)

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

    # print( type(test_data) , len(test_data))
    if config['fl_opt']['multi_scale'] and config['use_resize_data']:
        train_data_all_mid, train_label_all_mid, _ = \
            ImbalanceSpeech_clean("training", imb_ratio, root,  level = 1) # mid

        train_data_all_small, train_label_all_small, _ = \
            ImbalanceSpeech_clean("training", imb_ratio, root,  level = 0) # small 
        
        test_data_all_mid, test_label_all_mid, _ = \
            ImbalanceSpeech_clean("testing", imb_ratio, root,  level = 1) # mid
        
        test_data_all_small, test_label_all_small, _ = \
            ImbalanceSpeech_clean("testing", imb_ratio, root,  level = 0) # small

        for client_i in range(client_num):
            for j in idx_per_client[client_i]:
                per_client_data[client_i].append(train_data_all_mid[j]) 
                per_client_label[client_i].append(train_label_all_mid[j]) 
    
        for client_i in range(client_num):
            for j in idx_per_client[client_i]:
                per_client_data[client_i].append(train_data_all_small[j]) 
                per_client_label[client_i].append(train_label_all_small[j]) 

        test_data += test_data_all_mid
        test_label += test_label_all_mid
        test_data += test_data_all_small
        test_label += test_label_all_small


    return per_client_data, per_client_label, \
        test_data, test_label, val_data, val_label, \
        cls_per_client, num_per_cls_per_client, train_num_per_cls


def ImbalanceSpeech_clean(mode, imbalance_ratio, root, imb_type='exp', test_imb_ratio=None, reverse=False, config = None, level = 2):

    """
    mode: training, testing, validation  
    imbalance_ratio: only for training
    root: data folder  

    Important paramters
    ---
    Return: (all are lists)
    data: [a, b, ...], a and b are tensor of 1*32*32
    labels: [a, b, ...], a and b are scalar
    """
    
    cls_num = 35
    
    if mode == 'training':
        if level == 0 :
            data_file_path = "./data/training_small_data.pt"
            target_file_path = "./data/training_small_target.pt"
        elif level == 1:
            data_file_path = "./data/training_mid_data.pt"
            target_file_path = "./data/training_mid_target.pt"
        elif level == 2:
            data_file_path = f"./data/{mode}_data.pt"
            target_file_path = f"./data/{mode}_target.pt"

    elif mode == 'validation' and  level == 2:
        data_file_path = f"./data/{mode}_data.pt"
        target_file_path = f"./data/{mode}_target.pt"

    elif mode == 'testing':
        if level == 0 :
            data_file_path = "./data/testing_small_data.pt"
            target_file_path = "./data/testing_small_target.pt"
        elif level == 1:
            data_file_path = "./data/testing_mid_data.pt"
            target_file_path = "./data/testing_mid_target.pt"
        elif level == 2:
            data_file_path = f"./data/{mode}_data.pt"
            target_file_path = f"./data/{mode}_target.pt"
    
    else:
        raise RuntimeError


    if os.path.exists(data_file_path) is False:
        raise RuntimeError

    data = torch.load(data_file_path)
    targets = torch.load(target_file_path)

    # obtain the distribution
    num_per_cls = [0 for i in range(cls_num)]
    for label in targets:
        num_per_cls[label] += 1
    print(mode, "original num_per_cls:", num_per_cls)

    # change the distribution to long-tail
    if mode == "training":
        selected_num_per_cls = get_num_per_cls(num_per_cls, imb_type, imbalance_ratio, reverse=reverse)
        data, targets = gen_imbalanced_data(data, targets, selected_num_per_cls) # In Line 282
        print("selected num_per_cls:", mode, selected_num_per_cls)
    else:
        selected_num_per_cls = num_per_cls

    # adjust the data according to max/min/mean ~ (-68, 51, -18)
    min, max, mean = torch.min(data), torch.max(data), torch.mean(data)
    print("min/max/mean:", min, max, mean)
    data = (data + 9)/61
    # print("data/target shape:", data.shape, targets.shape, "\n")

    # change data and targets into list
    data_list, targets_list = [], []
    for i, j in zip(data, targets):
        data_list.append(i)
        targets_list.append(int(j))

    return data_list, targets_list, selected_num_per_cls

def get_num_per_cls(num_per_cls, imb_type, imb_factor, reverse=False):
        cls_num = len(num_per_cls)
        img_max = 3000      # manually defined
        img_min = min(num_per_cls)

        selected_num_per_cls = []
        if imb_factor == 1:
            selected_num_per_cls = [img_min for i in range(cls_num)]

        elif imb_factor <= 0.1:
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    if reverse:
                        num = img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    else:
                        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    selected_num_per_cls.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    selected_num_per_cls.append(int(img_max * imb_factor))
            else:
                selected_num_per_cls.extend([int(img_max)] * cls_num)
        return selected_num_per_cls


def gen_imbalanced_data(data, targets, selected_num_per_cls):
    """
    self.data: numpy.array
    self.targets: list
    """
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, selected_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        # print(the_class, the_img_num, idx)
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    data = torch.cat(new_data, 0)        # n*32*32*3
    targets = torch.tensor(new_targets)   # list of length n
    return data, targets



############## IMU ############
def IMU_FL(root, config):
    """
    Divide IMU dataset into small parts for FL.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return:
    ---
    per_client_data, per_client_label: list of lists
    test_data, test_label: both are lists
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]  # 5
    imb_ratio = config["dataset"]["imb_ratio"]  # 0.01, 0.1, 1

    # training

    # An example: 
    # train_data_all: [12406, 3, 32, 32]; train_data_all -> List; train_label_all: 12406;
    # train_num_per_cls: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    train_data_all, train_label_all, train_num_per_cls = load_IMU(
        root, train=True
    )

    # test
    test_data, test_label, test_num_per_cls = load_IMU(
        root, train=False
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


def load_IMU(root, train):
    """
    Load IMU dataset into memory
    """
    import pickle

    if train:
        with open(f'{root}/train_x_img.pkl', 'rb') as handle:
            train_x_img = np.concatenate(pickle.load(handle), axis=0)   # (23003, 32, 32, 3)
            # train_x_img = np.transpose(train_x_img, (0,3,1,2))
            # print(train_x_img.shape)
        with open(f'{root}/train_y.pkl', 'rb') as handle:
            train_y = np.concatenate(pickle.load(handle), axis=0)    # (23003,)
            # print(train_y.shape)
        num_per_cls = [0,0,0,0,0]
        for i in train_y:
            num_per_cls[int(i)] += 1
        return train_x_img, train_y, num_per_cls

    else:
        with open(f'{root}/test_x_img.pkl', 'rb') as handle:
            test_x_img = np.concatenate(pickle.load(handle), axis=0)   # (9932, 32, 32, 3)
            # test_x_img = np.transpose(test_x_img, (0,3,1,2))
            # print(test_x_img.shape)
        with open(f'{root}/test_y.pkl', 'rb') as handle:
            test_y = np.concatenate(pickle.load(handle), axis=0)    # (9932,)
            # print(test_y.shape)
        num_per_cls = [0,0,0,0,0]
        for i in test_y:
            num_per_cls[int(i)] += 1
        return test_x_img, test_y, num_per_cls


def Statefarm_FL(root, config):
    """
    Divide IMU dataset into small parts for FL.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return:
    ---
    per_client_data, per_client_label: list of lists
    test_data, test_label: both are lists
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]  # 5
    imb_ratio = config["dataset"]["imb_ratio"]  # 0.01, 0.1, 1

    # training

    # An example: 
    # train_data_all: [12406, 3, 32, 32]; train_data_all -> List; train_label_all: 12406;
    # train_num_per_cls: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    # all of data are <class 'numpy.ndarray'>
     
    users_train, train_data_all, train_label_all, train_num_per_cls = load_statefarm(
        root, train=True
    )

    # test
    _, test_data, test_label, test_num_per_cls = load_statefarm(
        root, train=False
    )

    train_data_all, train_label_all,  test_data, test_label = train_data_all.numpy(), train_label_all.numpy(), test_data.numpy(), test_label.numpy()
    # generate per-client FL data
    if config['fl_opt']['num_clients'] == 26 and config['others']['real']:

        # print(users_train)
        idx_per_client, cls_per_client, num_per_cls_per_client \
            = gen_fl_data_new(train_label_all, config, users_train)

    if config['fl_opt']['num_clients'] == 10 and config['others']['real']:

        idx_per_client, cls_per_client, num_per_cls_per_client = gen_fl_data(train_label_all, train_num_per_cls, config)
        idx_per_client_tmp = change_to_real(idx_per_client, train_label_all, users_train)
        idx_per_client = idx_per_client_tmp

    else:
        # reuse cifar framework
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


def load_statefarm(root, train):
    """
    - Load statefarm dataset into memory
    - Only use training set and split into 8 : 2
    - spliting data according to the users, 
    - If the client is 26, we use the true data distribution;
    - If not, we use the dir distribution;
    """
    import pickle

    if train:
        with open(f'{root}/trainset.pkl', 'rb') as handle:
            (users, train_x, train_y) = pickle.load(handle) # torch.Size([17948, 3, 256, 256])         
        np_train_y = train_y.numpy()

        num_per_cls = [0 for i in range(10)]
        for i in np_train_y:
            num_per_cls[int(i)] += 1
        return users, train_x, train_y, num_per_cls

    else:
        with open(f'{root}/testset.pkl', 'rb') as handle:
            (users, test_x, test_y) = pickle.load(handle) # torch.Size([4476, 3, 256, 256])
        
        np_test_y = test_y.numpy()
        num_per_cls = [0 for i in range(10)]
        for i in np_test_y:
            num_per_cls[int(i)] += 1
        return users, test_x, test_y, num_per_cls


def depth_FL(root, config):

    """
    Divide IMU dataset into small parts for FL.
    mode: base, novel or all classes
    shot_num: the number of n (n-shot)
    Return:
    ---
    per_client_data, per_client_label: list of lists
    test_data, test_label: both are lists
    """
    # shot_num = config['dataset']['shot']
    # assert shot_num != 0

    num_classes = config["networks"]["classifier"]["params"]["num_classes"]  # 5
    imb_ratio = config["dataset"]["imb_ratio"]  # 0.01, 0.1, 1

    # training

    # An example: 
    # train_data_all: [12406, 3, 32, 32]; train_data_all -> List; train_label_all: 12406;
    # train_num_per_cls: [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
    # all of data are <class 'numpy.ndarray'>
     
    users_train, train_data_all, train_label_all, train_num_per_cls = load_depth( root, train=True)

    # test
    _, test_data, test_label, test_num_per_cls = load_depth(root, train=False)

    train_data_all, train_label_all,  test_data, test_label = train_data_all.numpy(), train_label_all.numpy(), test_data.numpy(), test_label.numpy()
    # generate per-client FL data
    if config['fl_opt']['num_clients'] == 30 and config['others']['real'] :
        
        print("==> " + str(len(users_train)))
        print("==> " + str(np.unique(users_train)))

        # print(users_train)
        idx_per_client, cls_per_client, num_per_cls_per_client \
            = gen_fl_data_new(train_label_all, config, users_train)

    else:
        # reuse cifar framework

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



def load_depth(root, train):
    """
    - Load statefarm dataset into memory
    - Only use training set and split into 7 : 3
    - spliting data according to the users, 
    - If the client is 26, we use the true data distribution;
    - If not, we use the dir distribution;
    """
    import pickle

    if train:
        with open(f'{root}/trainset.pkl', 'rb') as handle:
            (users, train_x, train_y) = pickle.load(handle) # torch.Size([])         
        
        print(train_x.shape)
        np_train_y = train_y.numpy()

        num_per_cls = [0 for i in range(14)]
        for i in np_train_y:
            num_per_cls[int(i)] += 1
        return users, train_x, train_y, num_per_cls

    else:
        with open(f'{root}/testset.pkl', 'rb') as handle:
            (users, test_x, test_y) = pickle.load(handle) # torch.Size([4476, 3, 256, 256])
        
        np_test_y = test_y.numpy()
        num_per_cls = [0 for i in range(14)]
        for i in np_test_y:
            num_per_cls[int(i)] += 1
        return users, test_x, test_y, num_per_cls


def change_to_real(idx_per_client, train_label_all, users_train = None):
    """
    Note that train_label_all and user_train are unified order in statefarm datasets
    ---
    Argu:
        - idx_per_client: original distribution based on Non-IID data;
        - train_label_all: object of a class inheriting from torch.utils.data.Dataset 
            Or a list pre-stored in the RAM.
        - config: configuration dictionary
        - users_train: users, to mapping to the real distribution
    Return:
        - idx_per_client_tmp: list. The i^th item is the img idx of the training set for client i
    """  
    # 直接根据idx_per_client，拿到当前client的数据分布；
    # 依据数据分布，与user_train, 拿到真实的数据分布

    idx_per_client_tmp = []

    # print(type(users_train), users_train[:5])
    user_name_unique, user_number = np.unique(users_train, return_counts=True)
    # print(unique_1, number)


    for client_idx in range(len(idx_per_client)):         # client_idx in [0,1,2,3,...]

        idxs = idx_per_client[client_idx]
        # cal the distribution
        cls, counts = np.unique(np.array(train_label_all)[idxs], return_counts=True)


        user_name_in_loop = user_name_unique[client_idx]
        user_client_idx_where = np.where(np.array(users_train) == user_name_in_loop ) # find idx 
        user_client_idx = user_client_idx_where[0]

        for cls_item_idx in range(len(cls)): # 要第几类
            all_data_in_train_set_where = np.where(np.array(train_label_all) == cls[cls_item_idx]) # 找到所有这一类的索引在train_label_all中；（1）
            all_data_in_train_set = all_data_in_train_set_where[0]
            # print(len(all_data_in_train_set))

            idx_tmp = []
            for data_idx in all_data_in_train_set: # 要用所有的索引和user做交集 （2）
            
                if data_idx in user_client_idx and len(idx_tmp) < counts[cls_item_idx]: #如果长度没有超过我要的数目并且数据在这个client上我才要 （3）
                    idx_tmp.append(data_idx)
                

            idx_per_client_tmp.append(idx_tmp)


    return idx_per_client_tmp 