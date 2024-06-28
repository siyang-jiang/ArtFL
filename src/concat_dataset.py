
import copy, os
import numpy as np
import pickle
import random

def open_dataset(path, mode):
    data_in = open(path, "rb")
    data = pickle.load(data_in)
    image, label = data["image_data"], data["class_dict"]
    if mode == "train":
        image = image.reshape([64, 600, 84, 84, 3])
    elif mode == "val":
        image = image.reshape([16, 600, 84, 84, 3])
    elif mode == "test":
        image = image.reshape([20, 600, 84, 84, 3])
    else:
        raise ValueError("Mode Error")
    return image, label

def concat_mini_imagenet(train_path, val_path, test_path):
    train_image, train_label = open_dataset(train_path, mode = "train")
    val_image, val_label = open_dataset(val_path, mode = "val") 
    test_image, test_label = open_dataset(test_path, mode = "test")    

    data_image = np.concatenate([train_image, val_image, test_image], axis = 0)
    data_label = {}
    labels = [train_label, val_label, test_label]
    idx = 0
    for item in labels:
        for key in item.keys():
            data_label[key] = [idx] * 600
            idx += 1
    
    print(" shape of data in mini_imagenet:" + str( data_image.shape), ) 
    return data_image, data_label

if __name__ == "__main__":

####################
# Download the dataset from  https://www.kaggle.com/whitemoon/miniimagenet
# Use the script to concat the whole dataset
####################

    train_path = "/home/zhangxianghui/code/ExtremeFL/data/exFL/mini-imagenet/mini-imagenet-cache-train.pkl"
    val_path = "/home/zhangxianghui/code/ExtremeFL/data/exFL/mini-imagenet/mini-imagenet-cache-val.pkl"
    test_path = "/home/zhangxianghui/code/ExtremeFL/data/exFL/mini-imagenet/mini-imagenet-cache-test.pkl"
    store_path = "/home/zhangxianghui/code/ExtremeFL/data/exFL/mini-imagenet/"

    # train_path = "/ssd/syjiang/data/exFL/mini-imagenet/mini-imagenet-cache-train.pkl"
    # val_path = "/ssd/syjiang/data/exFL/mini-imagenet/mini-imagenet-cache-val.pkl"
    # test_path = "/ssd/syjiang/data/exFL/mini-imagenet/mini-imagenet-cache-test.pkl"
    # store_path = "/ssd/syjiang/data/exFL/mini-imagenet/"

    image, label = concat_mini_imagenet(train_path, val_path, test_path)

    with open(store_path+'mini_imagenet_data.pickle', 'wb') as f:
        pickle.dump(image, f)
    
    with open(store_path+'mini_imagenet_label.pickle', 'wb') as f:
        pickle.dump(label, f)  

    print("Already Store in " + store_path)


