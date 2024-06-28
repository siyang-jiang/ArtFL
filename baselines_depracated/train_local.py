import copy, os
import numpy as np
import pprint
import argparse
import yaml
import shutil 
import pickle
from pathlib import Path

import torch
torch.backends.cudnn.benchmark = True 
from torch.utils.tensorboard import SummaryWriter

from utils.misc import update_config, deterministic
from utils.logger import Logger, print_write, write_summary
from utils import utils 
from data import dataloader
from models.utils import *
# from fed import Fed_client, Fed_server
from fed_branch import Fed_client, Fed_server

from dataloader import LoaderCifar
from data.dataloader import mini_ImageNet_FL, tiny_miniImageNet_FL, Speech_FL, IMU_FL, Statefarm_FL, depth_FL

"""
Script for local training.
"""


if __name__ == '__main__':
    

    """
    Parameters
    """
    # important parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="config/cifar10/local.yaml", type=str)
    parser.add_argument("--exp_name", default="local", type=str, help="exp name")
    parser.add_argument("--work_dir", default="./exp_results", type=str, help="output dir")

    # unused params
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu_idx', default=0, type=int)
    # optional params
    # parser.add_argument('--tao_ratio', type=float, default=4, choices=[0.5, 1, 2, 4])
    # parser.add_argument("--non_iidness", default=1, type=int, help="non-iid degree of distributed data")
    # parser.add_argument('--imb_ratio', type=float, default=1, choices=[0.01, 0.05, 0.1, 1])
    
    args = parser.parse_args()

    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = utils.combine_all_config(args)
    # print(config)

    # random seed
    if config["seed"] is not None:
        deterministic(config["seed"])

    root_all, data_root_dict = utils.GetDatasetPath(config)
    
    # config and logger
    log_dir = f'{args.work_dir}/{args.exp_name}'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)  
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # with open(args.cfg) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # config = update_config(config, args, log_dir)    # let args overwite YAML config

    # logger
    logger = Logger(log_dir)
    logger.log_cfg(config)  # save cfg
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # tensorboard
    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")

    # prepare dataset
    dataset = config["dataset"]["name"]
    root_all, data_root_dict = utils.GetDatasetPath(config)
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

    print_write([cls_per_client], log_file)
    print_write([train_num_per_cls], log_file)
    print_write([num_per_cls_per_client], log_file)
    print(len(per_client_label[0]))


    """"
    FL config setup 
    """    
    # assign devices
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]['num_classes']

    gpu_list = [1, 2, 3]
    if len(gpu_list) > torch.cuda.device_count(): 
        raise RuntimeError
    gpu_idx = [gpu_list[i % len(gpu_list)] for i in range(num_client)] 
    config.update({"device": torch.device('cpu' if torch.cuda.is_available() else 'cpu')})   # init_network
    config.update({"device_client": [torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in gpu_idx]})    # client device\
    print(f"gpu of clients: {gpu_idx}")

    # init model and criterion on cpu
    network = init_models(config)   # print(network["feat_model"])
    criterion = init_criterion(config)
    
    # multi-process setup
    import multiprocessing as mp 
    torch.multiprocessing.set_sharing_strategy('file_system')
    if torch.cuda.is_available():
        mp.set_start_method('forkserver', force=True)
    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    state_dict_list = mp.Manager().list([None for i in range(num_client)])
    eval_result_list = mp.Manager().list([None for i in range(num_client)])
    # size probability
    size_prob = mp.Manager().list([[1/3, 1/3, 1/3] for i in range(num_client)])     
    # Computation number of whole progress. 
    size_accounter = mp.Manager().list([[0, 0, 0] for i in range(num_client)]) 

    # FL class for servrer
    fed = Fed_server(
        network, criterion, config, per_client_data, per_client_label, training_num_per_cls, 
        test_data, test_label, state_list, state_dict_list
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
            state_dict_list, 
            idx=i, 
            size_prob = size_prob,
            size_accounter = size_accounter,
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
        for round_i in range(1):

            # select users
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False)
            # select classes and num_of_samples per class
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write([f'\n Round: {round_i}, selected clients: {selected_idx}'], log_file)
            # print_write([f'selected cls: {set(selected_cls)}'], log_file)

            # train and evalute
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)
            all_results = fed.evaluate_local()
            # all_results, shape：(4, num_client, num_cls), 4 for (train_loss, train_acc, test_loss, test_acc)

            # mean of every client
            train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls = np.mean(all_results, axis=1)   
            print(train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls)
            test_acc_array = all_results[3]     # shape (num_client, num_cls)
            train_loss = train_loss_per_cls.mean()
            train_acc = train_acc_per_cls.mean()

            # testset (e.g., Cifar-100) is equally distributed among classes
            test_loss_mean = test_loss_per_cls.mean()
            test_acc_mean = test_acc_per_cls.mean()

            # logging
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(fed.losses_cls)[selected_idx].mean()
            np.set_printoptions(precision=3)
            print_write(["cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc: ",\
                [cls_loss, kd_loss, train_loss, test_loss_mean, train_acc, test_acc_mean]], log_file)
            print_write(["per_cls_acc (train): ", train_acc_per_cls], log_file)    
            print_write(["per_cls_acc: ", test_acc_per_cls], log_file)

            write_summary(
                tensorboard, split='train', step=round_i, kd_loss=kd_loss, 
                cls_loss=cls_loss, loss=train_loss, acc=train_acc, 
                cls0_acc=train_acc_per_cls[0], cls2_acc=train_acc_per_cls[1], cls3_acc=train_acc_per_cls[2],
                cls4_acc=train_acc_per_cls[3], cls5_acc=train_acc_per_cls[4], cls6_acc=train_acc_per_cls[5])
            write_summary(
                tensorboard, split='val', step=round_i, loss=test_loss_mean, acc=test_acc_mean, 
                cls0_acc=test_acc_per_cls[0], cls2_acc=test_acc_per_cls[1], cls3_acc=test_acc_per_cls[2],
                cls4_acc=test_acc_per_cls[3], cls5_acc=test_acc_per_cls[4], cls6_acc=test_acc_per_cls[5]
                )

            # save ckpts
            if test_acc_mean > best_acc:
                ckpt = {'round_i': round_i, 'model': fed.server_network}
                ckpt_name = f"{log_dir}/best.pth"
                acc_array_name = f"{log_dir}/acc_array.csv"  # perclient_percls accuracy array
                torch.save(ckpt, ckpt_name)
                np.savetxt(acc_array_name, test_acc_array, delimiter = ',')
                best_acc = test_acc_mean
                print_write([f"best round: {round_i}, accuracy: {test_acc_mean*100}"], log_file)
                # del ckpt

    # Currently do not support test mode. Use `evaluate.py` instead.
    else:
        pass

    for i in process_list:
        p.join()
