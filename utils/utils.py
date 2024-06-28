import yaml
import os.path as path
import collections
import sys
import torch
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
import re
import os
import math
import time
import csv
import numpy as np
import argparse
import sys
sys.path.append('../')


# support 1e-4
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


# access dict by obj.xxx
class DotDict(dict):
    """
    a.b.c
    >>>data = {
    ...    'api': '/api',
    ...    'data': {
    ...        'api': '/data/api'
    ...    }
    ...}
    >>>obj = DotDict(data)
    >>>obj.api
    '/api'
    >>>obj.data
    {'api': '/data/api'}
    >>>obj.data.api
    '/data/api'
    """

    def __init__(self, data_map=None):
        super(DotDict, self).__init__(data_map)
        if isinstance(data_map, dict):
            for k, v in data_map.items():
                if not isinstance(v, dict):
                    self[k] = v
                else:
                    self.__setattr__(k, DotDict(v))

    def __getattr__(self, attr):
        return self.get(attr, False)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

# read yaml config

def read_config(config_name):
    # with open(path.join("config", "{}.yaml".format(config_name)), "r") as f:
    with open(config_name, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


# for combining config dict
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# parse command to dict
def parse_command(params):
    params_cp = vars(params)
    return  params_cp, params


def combine_all_config(params=None):
    """
    read default.yam, your_config.yaml and command line key+vals
    """
    print()
    if params is None:
        params = sys.argv[1:]

    default_config = read_config(path.join("config", "{}.yaml".format('default')))
    command_config, params = parse_command(params) # update the config by config name

    if 'cfg' not in command_config:
        assert False, "please specify your config name. (name=xxx)"

    yaml_config = read_config(command_config['cfg'])

    # combine dict
    default_config = recursive_dict_update(default_config, yaml_config)
    default_config = recursive_dict_update(default_config, command_config)
    # config = DotDict(default_config)
    # return config
    return  default_config

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/fedavg.yaml", type=str)
    args = parser.parse_args()
    return args

def GetDatasetPath(config):

    if config["user"] == "User1":
        root = "/ssd/syjiang/data/exFL/mini-imagenet/"
        data_root_dict = {
        "Lidar_HAR": "./data/Lidar-HAR-dataset/avgray_image/",
        "CUB": "/data/xian/CUB_200_2011",
        "CIFAR100": "/media/disk3/syjiang/cifar_100/",
        "CIFAR10":  "~/Datasets/",
        "mini": "/ssd/syjiang/data/exFL/mini-imagenet/",
        "tiny": "/media/disk3/syjiang/tiny-imagenet-200",
        "Speech":"/media/disk3/syjiang/",
        "imu": "/media/disk3/syjiang/large_scale_HARBox",
        "statefarm": "/media/disk3/syjiang/state_farm",
        "depth": "/media/disk3/syjiang/depth"
    }
    root_all = {}
    
    root = os.path.join(root, str(config["seed"]))
    root_all['train_image'] = root + "/" + "train_image.pickle"
    root_all['train_label'] = root + "/" + "train_label.pickle"
    root_all['val_image'] = root + "/" + "val_image.pickle"
    root_all['val_label'] = root + "/" + "val_label.pickle"
    root_all['test_image'] = root + "/" + "test_image.pickle"
    root_all['test_label'] = root + "/" + "test_label.pickle"
    return root_all, data_root_dict

def set_device(config):

    # print(config)
    GPU_NUM = torch.cuda.device_count()
    if config['user'] == "User1":
        if config['dataset']['name'] in ['tiny', 'CIFAR100', 'statefarm', 'depth']:
            gpu_list = [i % GPU_NUM  for i in range(config["fl_opt"]["num_clients"])]
        else:
            gpu_list = [0,0,0,0,0,0,0,0,0,0]
    else:
        raise RuntimeError

    gpu_idx = [gpu_list[i % len(gpu_list)] for i in range(config["fl_opt"]["num_clients"])]
    print(f"gpu of clients: {gpu_idx}")

    # init_network
    config.update({"device": \
        torch.device("cpu" if torch.cuda.is_available() else "cpu")})  

    # client device
    config.update({"device_client": \
        [torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu") for i in gpu_idx]}) 
    
    return config


