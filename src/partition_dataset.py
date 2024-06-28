# ulimit -n 64000 

import copy, os
import numpy as np
import pickle
import random
import torch

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        # image = image.reshape([64, 600, 84, 84, 3])
    f.close()
    return data

def partiton(data, label):
    # data

    data = torch.tensor(data)
    left = int(np.floor(train_ratio * len(partition_idx)))
    right = int(np.floor(train_ratio * len(partition_idx))+np.floor(val_ratio * len(partition_idx)))
    train_set = data[:, :left, ::]
    val_set = data[:, left:right, ::]
    test_set = data[:, right:, ::]
    # train_set, val_set, test_set = torch.tensor_split(data, (left, right), 1)
    # print(train_set.shape, val_set.shape, test_set.shape)

    # label
    # repreduce the several times 
    train_label = {}
    val_label = {}
    test_label = {}

    idx = 0
    for key in label.keys():
        train_label[key] = [idx] * left
        val_label[key] = [idx] * (right - left)
        test_label[key] = [idx] * (len(partition_idx)- right)
        # print(left, (right - left), ((len(partition_idx) - right)))
        idx +=1

    
    return train_set, train_label, val_set, val_label, test_set, test_label

if __name__ == "__main__":
####################
#   After obtaining the whole dataset;
#   Split the dataset by different random seeds.
####################
    lucky_seed = 2021
    train_ratio, val_ratio = 0.7, 0.1
    random.seed(lucky_seed)

    partition_idx = [i for i in range(0,600)]
    random.shuffle(partition_idx)
    # print(partition_idx)
    train_idx = partition_idx[:int(np.floor(train_ratio * len(partition_idx)))]
    val_idx = partition_idx[len(train_idx):len(train_idx)+int(np.floor(val_ratio * len(partition_idx)))]
    test_idx = partition_idx[len(train_idx)+len(val_idx):]
    print(len(train_idx),len(val_idx), len(test_idx))

    # root = "/ssd/syjiang/data/exFL/mini-imagenet/"
    root = "/home/zhangxianghui/code/ExtremeFL/data/exFL/mini-imagenet"

    image_path = os.path.join(root,"mini_imagenet_data.pickle")
    label_path = os.path.join(root,"mini_imagenet_label.pickle")
    data = load_data(image_path)
    label = load_data(label_path)

    train_set, train_label, val_set, val_label, test_set, test_label = partiton(data, label)

    if not os.path.exists(os.path.join(root, str(lucky_seed)) ):
        os.mkdir( (os.path.join(root, str(lucky_seed)) ))
    
    store_path = os.path.join( root, str(lucky_seed))
    # print(store_path)
    with open( store_path+'/train_image.pickle', 'wb' ) as f:
        pickle.dump(train_set, f)

    with open(store_path+'/train_label.pickle', 'wb') as f:
        pickle.dump(train_label, f)

    with open(store_path+'/val_image.pickle', 'wb') as f:
        pickle.dump(val_set, f)

    with open(store_path+'/val_label.pickle', 'wb') as f:
        pickle.dump(val_label, f)

    with open(store_path+'/test_image.pickle', 'wb') as f:
        pickle.dump(test_set, f)

    with open(store_path+'/test_label.pickle', 'wb') as f:
        pickle.dump(test_label, f)








