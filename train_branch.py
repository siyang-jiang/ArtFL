# ulimit -n 64000 

import copy, os
import ctypes
import numpy as np
import argparse
import yaml
import shutil
# from functools import total_ordering
# from signal import raise_signal
# import pickle
# import pprint

from pathlib import Path

import torch

# torch.backends.cudnn.benchmark = True
from torch.utils.tensorboard import SummaryWriter

from utils.misc import update_config, deterministic
from utils.logger import Logger, print_write, write_summary
from utils import utils 
from data import dataloader
from models.utils import *
from fed_branch import Fed_client, Fed_server
from data.dataloader import mini_ImageNet_FL, tiny_miniImageNet_FL, Speech_FL, IMU_FL, Statefarm_FL, depth_FL
from collections import Counter
from dataloader import LoaderCifar
from tqdm import tqdm

# from multiprocessing.managers import AcquirerProxy, SyncManager


"""
Script for original federated training (train only a global model). 
Support different datasets/losses/samplings. 
"""

def computation_resource_computation(size_accounter):
    total_computation = 0
    for size_item in size_accounter:
        total_computation += size_item[0] * 1 + size_item[1] * 2.25 + size_item[2] * 4
    return total_computation
    



if __name__ == "__main__":

    """
    Parameters
    """

    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/cifar10/feddyn.yaml", type=str)
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

    log_dir = f"{config['metainfo']['work_dir']}/{config['metainfo']['exp_name']}_{config['dataset']['dirichlet']}_{config['fl_opt']['num_clients']}"

    print(log_dir)

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

    print_write( ["Config: =====> ", config], log_file, )

    # prepare dataset
    dataset = config["dataset"]["name"]
    data_root = data_root_dict[dataset]
    # print(dataset,data_root)
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
    elif dataset == "tiny":
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
        ) = tiny_miniImageNet_FL(data_root, config)
    elif dataset == "Speech":
        (
            per_client_data, 
            per_client_label, 
            test_data, 
            test_label, 
            val_data, 
            val_label, 
            cls_per_client, 
            num_per_cls_per_client, 
            train_num_per_cls,
        ) = Speech_FL(data_root, config)
    elif dataset == "imu":
        (
            per_client_data,
            per_client_label,
            test_data,
            test_label,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
        ) = IMU_FL(data_root, config)
    elif dataset == "statefarm":
        (
            per_client_data,
            per_client_label,
            test_data,
            test_label,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
        ) = Statefarm_FL(data_root, config) 
    elif dataset == "depth":
        (
            per_client_data,
            per_client_label,
            test_data,
            test_label,
            cls_per_client,
            num_per_cls_per_client,
            train_num_per_cls,
        ) = depth_FL(data_root, config) 

    training_num_per_cls = np.array([len(i) for i in per_client_label])

    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]
    config = utils.set_device(config)

    # init/load model and criterion on cpu
    network = init_models(config) 
    criterion = init_criterion(config)

    import multiprocessing as mp
    if torch.cuda.is_available():
        mp.set_start_method('forkserver', force=True)

    size_list = config["hetero_size"]["sizes"]
    size_idx_list, number_per_size = dataloader.size_sampling(num_client, config["hetero_size"]["level"])
    size_per_client = np.array(size_list)[size_idx_list]
    size_percentage = [sum(size_idx_list==0), sum(size_idx_list==1), sum(size_idx_list==2)]
    
    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    model_list = mp.Manager().list([None for i in range(num_client)])
    eval_result_list = mp.Manager().list([None for i in range(num_client)])     # loss every round every client
    size_prob = mp.Manager().list([[1/3, 1/3, 1/3] for i in range(num_client)])     # size probability
    
    # lr for each client, Shape: [num_client,]
    lr_collect = mp.Manager().list([0 for i in range(num_client)]) 
    # Computation number of whole progress. 
    size_accounter = mp.Manager().list([[0, 0, 0] for i in range(num_client)]) 
    # Calculate Time of whole progress.
    time_record = mp.Manager().list([1 for i in range(num_client)]) 
    waiting_time_record =  mp.Manager().list([1 for i in range(num_client)]) 


    # Passing Drop Probliity
    drop_prob = mp.Manager().list([0 for i in range(num_client)]) # Each Client will recieve a drop probality.
    round_conter = mp.Manager().Value(ctypes.c_int, 0)


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
        model_list,     # a.k.a, state_dict_list
        eval_result_list = eval_result_list,
        size_per_client = size_per_client,   # for multi-branch case
        size_prob = size_prob,
        lr_collect = lr_collect,
        size_accounter = size_accounter,
        time_record = time_record,
        drop_prob = drop_prob,
        round_conter = round_conter,
        waiting_time_record = waiting_time_record    
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
            model_list,     # a.k.a, state_dict_list
            eval_result_list,
            idx=i,
            size_i=size_per_client[i],
            size_prob = size_prob,
            lr_collect = lr_collect,
            size_accounter = size_accounter,
            time_record = time_record,
            drop_prob = drop_prob,
            round_conter = round_conter,
            waiting_time_record = waiting_time_record
        )

        p.daemon = True
        p.start()
        process_list.append(p)

    """"
    FL starts 
    """
    # training mode
    waiting_time = 0
    if not config['test']:
        best_acc = 0

        # FL rounds
        for round_i in tqdm(range(config["fl_opt"]["rounds"])):   

            # select users
            round_conter.value = round_i
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write([f"\n\n Round: {round_i}, selected clients: {selected_idx}"], log_file)
            
            # FL
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)
            waiting_time += max(fed.waiting_time_record) - min(fed.waiting_time_record)

            fed.multi_scale_update()
            train_loss_array, train_acc_array, test_loss_array, test_acc_array = \
                fed.evaluate_global_size_hetero(skip_train=True)
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(eval_result_list).mean()


            ##################
            #### logging #####
            np.set_printoptions(precision=3)

            # per-size accuracy
            if config["hetero_size"]["train_hetero"]:       
                train_loss_list = train_loss_array.mean(1)
                train_acc_list = train_acc_array.mean(1)
                test_loss_list = test_loss_array.mean(1)
                test_acc_list = test_acc_array.mean(1)

                num_of_client = config["fl_opt"]["num_clients"]
                train_loss_mean = np.sum(train_loss_list*number_per_size) / num_of_client
                train_acc_mean = np.sum(train_acc_list*number_per_size) / num_of_client
                test_loss_mean = np.sum(test_loss_list*number_per_size) / num_of_client
                test_acc_mean = np.sum(test_acc_list*number_per_size) / num_of_client       
                size_accounter_tmp = fed.get_size_accounter()
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
            write_summary(
                tensorboard, split="Each_Size", step=round_i,
                acc_16=test_acc_list[0], acc_24=test_acc_list[1], acc_32=test_acc_list[2],
            )
            if test_acc_mean > best_acc:
                best_acc = test_acc_mean
                print_write(
                    [f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file
                )

    total_computation = computation_resource_computation(size_accounter)

    print_write(["total_computation: ", total_computation], log_file)
    print_write(["waiting_time: ", waiting_time / config['fl_opt']['rounds']], log_file)

    for i in process_list:
        p.terminate()

