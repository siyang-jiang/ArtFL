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

# torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from utils.misc import update_config, deterministic
from utils.logger import Logger, print_write, write_summary
from utils import utils 
from data import dataloader
from models.utils import *
from fed import Fed_client, Fed_server
from data.dataloader import mini_ImageNet_FL
from collections import Counter

from dataloader import LoaderCifar
from tqdm import tqdm
"""
Script for original federated training (train only a global model). 
Support different datasets/losses/samplings. 
"""

if __name__ == "__main__":

    """
    Parameters
    """

    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/lth.yaml", type=str)
    args = parser.parse_args()
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = utils.combine_all_config(args)
    # config = update_config(config, args, log_dir)  # let args overwite YAML config

    # random seed
    if config["seed"] is not None:
        deterministic(config["seed"])

    root_all, data_root_dict = utils.GetDatasetPath(config)

    # logger
    log_dir = f"{config['metainfo']['work_dir']}/{config['metainfo']['exp_name']}_{config['dataset']['dirichlet']}_try_lth"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # tensorboard
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    # prepare dataset
    dataset = config["dataset"]["name"]
    data_root = data_root_dict[dataset]
    if dataset in ["CIFAR100", "CIFAR10"]:
        if config["dataset"]["shot_few"] > 0:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
            ) = LoaderCifar.CIFAR_FL_mixed(data_root, config)
        else:
            (
                per_client_data,
                per_client_label,
                test_data,
                test_label,
                cls_per_client,
                num_per_cls_per_client,
                train_num_per_cls,
            ) = LoaderCifar.CIFAR_FL(data_root, config)
    elif dataset == "mini":
        (
            per_client_data,
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

    print_write([cls_per_client], log_file)
    print_write([train_num_per_cls], log_file)
    print_write([num_per_cls_per_client], log_file)

    """"
    FL config setup 
    """
    # assign devices
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]

    config = utils.set_device(config)
    # print(config["device_client"]); print(torch.cuda.is_available())

    # init/load model and criterion on cpu
    # network = init_models(config) 
    network = torch.load(config["metainfo"]["ckpt"], map_location=torch.device('cpu'))["model"]
    criterion = init_criterion(config)

    import multiprocessing as mp
    # torch.multiprocessing.set_sharing_strategy('file_system')
    if torch.cuda.is_available():
        # mp.set_start_method("spawn", force=True)
        mp.set_start_method('forkserver', force=True)

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
        p = Fed_client(
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
            idx=i,
        )
        p.daemon = True
        p.start()
        process_list.append(p)

    """"
    FL starts 
    """
    # training mode
    if not config['test']:
        best_acc = 0

        # FL rounds
        for round_i in tqdm(range(config["fl_opt"]["rounds"])):   

            ########################
            ##### select users #####
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            # print select clients and classes
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write([f"\n Round: {round_i}, selected clients: {selected_idx}"], log_file)
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)
            
            # LTH finetuning
            fed.local_train(selected_idx)
            train_loss_array, train_acc_array, test_loss_array, test_acc_array = fed.evaluate_local_size_hetero(fast=True)
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(fed.losses_cls)[selected_idx].mean()

            ###################
            ##### logging #####
            np.set_printoptions(precision=3)

            # per-size accuracy
            if config["hetero_size"]["train_hetero"]:       
                # size_percentage
                size_idx_list = dataloader.size_sampling(config["hetero_size"]["level"])
                size_percentage = [sum(size_idx_list==0), sum(size_idx_list==1), sum(size_idx_list==2)]
                # accuracy: mean-of-all-classes, per-size
                train_loss_list = train_loss_array.mean(1)
                train_acc_list = train_acc_array.mean(1)
                test_loss_list = test_loss_array.mean(1)
                test_acc_list = test_acc_array.mean(1)

                # weighted sum
                num_of_client = config["fl_opt"]["num_clients"]
                train_loss_mean = np.sum([train_loss_list]) / num_of_client
                train_acc_mean = np.sum([train_acc_list]) / num_of_client
                test_loss_mean = np.sum([test_loss_list]) / num_of_client
                test_acc_mean = np.sum([test_acc_list]) / num_of_client       
                print_write(
                    [
                        "cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc, test_acc_mean: ",
                        [cls_loss, kd_loss, train_loss_mean, test_loss_mean, train_acc_list, test_acc_list, test_acc_mean],
                    ],
                    log_file,
                )

            # per-cls accuracy
            else:       
                # mean-of-all-sizes, per-cls
                train_loss_list = train_loss_array.mean(0)
                train_acc_list = train_acc_array.mean(0)
                test_loss_list = test_loss_array.mean(0)
                test_acc_list = test_acc_array.mean(0)
                # mean-of-all-size and -of-all-cls
                train_loss_mean = train_loss_list.mean()
                train_acc_mean = train_acc_list.mean()
                test_loss_mean = test_loss_list.mean()
                test_acc_mean = test_acc_list.mean()
                print_write(["per_cls_acc (train): ", train_acc_list], log_file)
                print_write(["per_cls_acc: ", test_acc_list], log_file)

            # tensorboard
            write_summary(
                tensorboard, split="train", step=round_i, kd_loss=kd_loss, 
                cls_loss=cls_loss, loss=train_loss_mean, acc=train_acc_mean,
            )
            write_summary(
                tensorboard, split="val", step=round_i,
                loss=test_loss_mean, acc=test_acc_mean,
            )

            # save every ckpt
            """
            torch.save(
                {
                    "round_i": round_i,
                    "server_network": fed.server_network,
                    "client_network": fed.networks,
                    "train_acc_per_cls": train_acc, 
                    "test_acc_per_cls": test_acc_mean,
                },  f"{log_dir}/{round_i}.pth",
            )
            """

            # only save the best ckpts
            if test_acc_mean > best_acc:
                ckpt = {"round_i": round_i, "model": fed.server_network}
                ckpt_name = f"{log_dir}/best.pth"
                torch.save(ckpt, ckpt_name)
                best_acc = test_acc_mean
                print_write(
                    [f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file
                )
                del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    for i in process_list:
        p.join() 

