import copy, os
import numpy as np
import pprint
import argparse
import yaml
import shutil
import pickle
from pathlib import Path
from utils.train_helper import shot_acc, validate_one_model
import torch
from data.dataloader import mini_ImageNet_FL

torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from utils.misc import update_config, deterministic
from utils.sampling import gen_fl_data, gen_ptest_data
from utils.logger import Logger, print_write, write_summary
from data import dataloader
from models.utils import *
from utils.train_helper import shot_acc
from fed import Fed_client, Fed_server


"""
Script for original federated training (train only a global model). 
Support different datasets/losses/samplings. 
"""


def set_device(config):
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]
    # gpu_list = [1, 2, 3, 0][: torch.cuda.device_count()]
    gpu_list = [ 0 ][: torch.cuda.device_count()] # For SY 1

    # gpu_list = [1, 2, 3]
    # if len(gpu_list) > torch.cuda.device_count():
    #     raise RuntimeError
    gpu_idx = [gpu_list[i % len(gpu_list)] for i in range(num_client)]
    config.update(
        {"device": torch.device("cpu" if torch.cuda.is_available() else "cpu")}
    )  # init_network
    config.update(
        {
            "device_client": [
                torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
                for i in gpu_idx
            ]
        }
    )  # client device\
    print(f"gpu of clients: {gpu_idx}") # already

    return num_client, num_cls


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/fed_avg_mini.yaml", type=str)
    parser.add_argument("--exp_name", default="central1", type=str, help="exp name")
    parser.add_argument(
        "--non_iidness", default=1, type=int, help="non-iid degree of distributed data"
    )
    parser.add_argument("--tao_ratio", type=float, default= 1/420, choices=[0.5, 1, 2, 4])
    # optional params
    parser.add_argument("--seed", default=1, type=int, help="using fixed random seed")
    parser.add_argument("--work_dir", default="./runs_mini", type=str, help="output dir")
    # unused params
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()

    return args


def log_config(args):

    log_dir = f"{args.work_dir}/{args.exp_name}_{args.tao_ratio}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = update_config(config, args, log_dir)  # let args overwite YAML config

    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # print(config["dataset"]["tao_ratio"])
    # raise ValueError
    return log_dir, log_file, config


if __name__ == "__main__":

    # setting parameters
    args = get_args()
    log_dir, log_file, config = log_config(args)
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    root = "/ssd/syjiang/data/exFL/mini-imagenet/"
    # root = "/home/shuaixian/code/ExtremeFL/data/exFL/mini-imagenet"

    seed = 2021
    root = os.path.join(root, str(seed))
    root_all = {}
    root_all['train_image'] = root + "/" + "train_image.pickle"
    root_all['train_label'] = root + "/" + "train_label.pickle"
    root_all['val_image'] = root + "/" + "val_image.pickle"
    root_all['val_label'] = root + "/" + "val_label.pickle"
    root_all['test_image'] = root + "/" + "test_image.pickle"
    root_all['test_label'] = root + "/" + "test_label.pickle"

    (   per_client_data,
        per_client_label,
        val_data,
        val_label,
        test_data,
        test_label,
        cls_per_client,
        num_per_cls_per_client,
        train_num_per_cls,
    ) = mini_ImageNet_FL(root_all, config)

    _, _ = set_device(config)
    # gpu_list = [ 3 ][: torch.cuda.device_count()]

    config.update({"device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')})
    device = config["device"]
    # print(device)
    # raise  ValueError

    training_num_per_cls = np.array([len(i) for i in per_client_label])
    print_write([cls_per_client], log_file)

    # combined distributed dataset into centralized sets
    # train
    train_data, train_label = [], []
    for client_i in range(10):  # 10 client
        train_data.extend(per_client_data[client_i])
        train_label.extend(per_client_label[client_i])

    train_dataset = dataloader.local_client_dataset(train_data, train_label, config, phase="train", aug=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 128, shuffle = True,
        num_workers = 16, pin_memory=False)

    # test
    test_dataset = dataloader.local_client_dataset(test_data, test_label, config, phase="test")
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True, 
    #     num_workers = 16, pin_memory=True)

    # model, criterion, optimizers
    network = init_models(config)
    criterion = init_criterion(config)
    optimizer = init_optimizers(network, config)


    # training mode
    if not args.test:
        best_acc = 0

        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):
            
            for key in network.keys():
                network[key].train()     

            for (imgs, labels, indexs) in train_loader:

                # to device
                imgs = imgs.to(device)
                labels = labels.to(device)
                # forward
                feat = network['feat_model'](imgs)
                logits = network['classifier'](feat)
                # loss    
                loss, loss_cls, loss_kd = criterion(logits, labels)
                # backward
                for opt in optimizer.values():
                    opt.zero_grad()
                loss.backward()
                for opt in optimizer.values():
                    opt.step()
                # classifier L2-norm
                if network['classifier'].l2_norm:
                    network['classifier'].weight_norm()

            # evaluate
            for key in network.keys():
                network[key].eval()     
            train_loss_per_cls, train_acc_per_cls = validate_one_model(
                network, train_dataset, device, per_cls_acc=True) 
            test_loss_per_cls, test_acc_per_cls = validate_one_model(
                network, test_dataset, device, per_cls_acc=True) 

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()
            train_loss_mean = train_loss_per_cls.mean()
            train_acc_mean = train_acc_per_cls.mean()

            # logging
            print_write(["\n", "train_loss, train_acc_per_cls, test_loss, per_cls_acc, test_acc_mean: ",\
                train_loss_mean, train_acc_per_cls, test_loss_mean, test_acc_per_cls, test_acc_mean], log_file)
            write_summary(
                tensorboard, split='train', step=round_i, loss=train_loss_mean, acc=train_acc_mean)
            write_summary(
                tensorboard, split='val', step=round_i, loss=test_loss_mean, acc=test_acc_mean, 
                cls0_acc=test_acc_per_cls[0], cls2_acc=test_acc_per_cls[1], cls3_acc=test_acc_per_cls[2],
                cls4_acc=test_acc_per_cls[3], cls5_acc=test_acc_per_cls[4], cls6_acc=test_acc_per_cls[5]
                )

            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {'round_i': round_i, 'model': network}
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write([f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file)
                # del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass