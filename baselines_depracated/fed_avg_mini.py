# ulimit -n 64000 

import copy, os
import numpy as np
import pprint
import argparse
import yaml
import shutil
import pickle
from pathlib import Path

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
    gpu_list = [1, 2, 3, 0][: torch.cuda.device_count()]
    # gpu_list = [ 1, 3 ][: torch.cuda.device_count()] # For SY 1
    
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
    parser.add_argument("--exp_name", default="fedavg", type=str, help="exp name")
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
    # root = "/mnt/ssd/xianghui/ExtremeFL/data/exFL/mini-imagenet"

    seed = 2021
    root = os.path.join(root, str(seed))

    root_all = {}

    root_all['train_image'] = root + "/" + "train_image.pickle"
    root_all['train_label'] = root + "/" + "train_label.pickle"

    root_all['val_image'] = root + "/" + "val_image.pickle"
    root_all['val_label'] = root + "/" + "val_label.pickle"

    root_all['test_image'] = root + "/" + "test_image.pickle"
    root_all['test_label'] = root + "/" + "test_label.pickle"

    # set_train(root_all, config)
    # Training Progress
    # Eval on Mini-Imagenet
    # Eval on EuroSAT

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

    training_num_per_cls = np.array([len(i) for i in per_client_label])
    num_client, num_cls = set_device(config)

    network = init_models(config)  # print(network["feat_model"])
    criterion = init_criterion(config)

    # multi-process setup
    import multiprocessing as mp

   # torch.multiprocessing.set_sharing_strategy('file_system')   
    if torch.cuda.is_available():
        # mp.set_start_method('fork')
        mp.set_start_method("spawn")
        # mp.set_start_method('forkserver', force=True)

    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    model_list = mp.Manager().list([None for i in range(num_client)])

    # FL class for servrer
    fed = Fed_server(
        network,
        criterion,
        config,
        per_client_data,
        per_client_label,
        training_num_per_cls,
        test_data,
        test_label,
        state_list,
        model_list,
    )
    
    aggre_mode = config["fl_opt"]["aggregation"]
    frac = config["fl_opt"]["frac"]
    client_per_round = max(int(frac * fed.num_clients), 1)

    # FL class for clients
    for i in range(num_client):
        client_i_data = [j for j in per_client_data[i]]
        client_i_label = per_client_label[i] 
        p = Fed_client(
            network,
            criterion,
            config,
            client_i_data,
            client_i_label,
            training_num_per_cls,
            test_data,
            test_label,
            state_list,
            model_list,
            idx=i,
        )
        p.daemon = True
        p.start()
        process_list.append(p)
    
    """"
    FL starts 
    """

    # training mode
    if not args.test:
        best_acc = 0

        # FL rounds
        for round_i in range(config["fl_opt"]["rounds"]):

            # select users
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            # select classes and num_of_samples per class
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write(
                [f"\n Round: {round_i}, selected clients: {selected_idx}"], log_file
            )
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)

            # train and aggregate
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)

            # evaluate
            (
                train_loss_per_cls,
                train_acc_per_cls,
                test_loss_per_cls,
                test_acc_per_cls,
            ) = fed.evaluate_global()
            train_loss = train_loss_per_cls.mean()
            train_acc = train_acc_per_cls.mean()

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()

            # logging
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(fed.losses_cls)[selected_idx].mean()
            np.set_printoptions(precision=3)
            print_write(
                [
                    "cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc: ",
                    [
                        cls_loss,
                        kd_loss,
                        train_loss,
                        test_loss_mean,
                        train_acc,
                        test_acc_mean,
                    ],
                ],
                log_file,
            )
            print_write(["per_cls_acc (train): ", train_acc_per_cls], log_file)
            print_write(["per_cls_acc: ", test_acc_per_cls], log_file)

            write_summary(
                tensorboard,
                split="train",
                step=round_i,
                kd_loss=kd_loss,
                cls_loss=cls_loss,
                loss=train_loss,
                acc=train_acc,
                cls0_acc=train_acc_per_cls[0],
                cls2_acc=train_acc_per_cls[1],
                cls3_acc=train_acc_per_cls[2],
                cls4_acc=train_acc_per_cls[3],
                cls5_acc=train_acc_per_cls[4],
                cls6_acc=train_acc_per_cls[5],
            )
            write_summary(
                tensorboard,
                split="val",
                step=round_i,
                loss=test_loss_mean,
                acc=test_acc_mean,
                cls0_acc=test_acc_per_cls[0],
                cls2_acc=test_acc_per_cls[1],
                cls3_acc=test_acc_per_cls[2],
                cls4_acc=test_acc_per_cls[3],
                cls5_acc=test_acc_per_cls[4],
                cls6_acc=test_acc_per_cls[5],
            )
            torch.save(
                {
                    "round_i": round_i,
                    "server_network": fed.server_network,
                    "client_network": fed.networks,
                    "train_acc_per_cls": train_acc_per_cls,
                    "test_acc_per_cls": test_acc_per_cls,
                },
                f"{log_dir}/{round_i}.pth",
            )
            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {"round_i": round_i, "model": fed.server_network}
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write(
                    [f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file
                )
                # del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    for p in process_list:
        p.join() ### ？

    