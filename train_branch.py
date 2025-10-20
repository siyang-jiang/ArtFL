import copy
import os
import ctypes
import shutil
from pathlib import Path

import numpy as np
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.misc import update_config, deterministic
from utils.logger import Logger, print_write, write_summary
from utils import utils 
from data import dataloader
from models.utils import *
from fed_branch import Fed_client, Fed_server
from data.dataloader import (
    mini_ImageNet_FL, tiny_miniImageNet_FL, Speech_FL, 
    IMU_FL, Statefarm_FL, depth_FL
)
from dataloader import LoaderCifar


def compute_total_computation(size_accounter):
    """
    Calculate total computation cost based on size distribution.
    Cost weights: [1, 2.25, 4] for sizes [small, medium, large]
    """
    total_computation = 0
    cost_weights = [1, 2.25, 4]
    
    for size_item in size_accounter:
        for i, cost in enumerate(cost_weights):
            total_computation += size_item[i] * cost
    
    return total_computation


if __name__ == "__main__":
    # ============================================================
    # Configuration and Setup
    # ============================================================
    parser = argparse.ArgumentParser(description="ArtFL Federated Learning Training")
    parser.add_argument("--cfg", default="config/cifar10/feddyn.yaml", type=str, 
                       help="Path to configuration file")
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = utils.combine_all_config(args)

    # Set random seed for reproducibility
    if config["seed"] is not None:
        deterministic(config["seed"])

    root_all, data_root_dict = utils.GetDatasetPath(config)

    # ============================================================
    # Logger and Tensorboard Setup
    # ============================================================
    log_dir = f"{config['metainfo']['work_dir']}/{config['metainfo']['exp_name']}_{config['dataset']['dirichlet']}_{config['fl_opt']['num_clients']}"
    print(f"Experiment directory: {log_dir}")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir)
    logger.log_cfg(config)
    log_file = f"{log_dir}/log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    tensorboard = SummaryWriter(log_dir=f"{log_dir}/tensorboard")
    print_write(["Config: =====> ", config], log_file)

    # ============================================================
    # Dataset Preparation
    # ============================================================
    dataset = config["dataset"]["name"]
    data_root = data_root_dict[dataset]
    
    # Load dataset based on type
    if dataset in ["CIFAR100", "CIFAR10"]:
        if config["dataset"]["shot_few"] > 0:
            (per_client_data, per_client_label, test_data, test_label, 
             cls_per_client) = LoaderCifar.CIFAR_FL_mixed(data_root, config)
        else:
            (per_client_data, per_client_label, test_data, test_label, 
             cls_per_client, num_per_cls_per_client, 
             train_num_per_cls) = LoaderCifar.CIFAR_FL(data_root, config)
    elif dataset == "mini":
        (per_client_data, per_client_label, val_data, val_label, 
         test_data, test_label, cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = mini_ImageNet_FL(root_all, config)
    elif dataset == "tiny":
        (per_client_data, per_client_label, val_data, val_label, 
         test_data, test_label, cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = tiny_miniImageNet_FL(data_root, config)
    elif dataset == "Speech":
        (per_client_data, per_client_label, test_data, test_label, 
         val_data, val_label, cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = Speech_FL(data_root, config)
    elif dataset == "imu":
        (per_client_data, per_client_label, test_data, test_label, 
         cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = IMU_FL(data_root, config)
    elif dataset == "statefarm":
        (per_client_data, per_client_label, test_data, test_label, 
         cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = Statefarm_FL(data_root, config) 
    elif dataset == "depth":
        (per_client_data, per_client_label, test_data, test_label, 
         cls_per_client, num_per_cls_per_client, 
         train_num_per_cls) = depth_FL(data_root, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    training_num_per_cls = np.array([len(i) for i in per_client_label])
    num_client = config["fl_opt"]["num_clients"]
    num_cls = config["dataset"]["num_classes"]
    config = utils.set_device(config)

    # ============================================================
    # Model and Criterion Initialization
    # ============================================================
    network = init_models(config) 
    criterion = init_criterion(config)

    # Set multiprocessing method for CUDA
    import multiprocessing as mp
    if torch.cuda.is_available():
        mp.set_start_method('forkserver', force=True)

    # ============================================================
    # Client Size Sampling for Heterogeneous Training
    # ============================================================
    size_list = config["hetero_size"]["sizes"]
    size_idx_list, number_per_size = dataloader.size_sampling(
        num_client, config["hetero_size"]["level"]
    )
    size_per_client = np.array(size_list)[size_idx_list]
    size_percentage = [
        sum(size_idx_list == 0), 
        sum(size_idx_list == 1), 
        sum(size_idx_list == 2)
    ]
    
    # ============================================================
    # Shared Memory for Multi-Process Communication
    # ============================================================
    process_list = []
    state_list = mp.Manager().list([0 for i in range(num_client)])
    model_list = mp.Manager().list([None for i in range(num_client)])
    eval_result_list = mp.Manager().list([None for i in range(num_client)])
    size_prob = mp.Manager().list([[1/3, 1/3, 1/3] for i in range(num_client)])
    lr_collect = mp.Manager().list([0 for i in range(num_client)]) 
    size_accounter = mp.Manager().list([[0, 0, 0] for i in range(num_client)]) 
    time_record = mp.Manager().list([1 for i in range(num_client)]) 
    waiting_time_record = mp.Manager().list([1 for i in range(num_client)]) 
    drop_prob = mp.Manager().list([0 for i in range(num_client)])
    round_conter = mp.Manager().Value(ctypes.c_int, 0)

    # ============================================================
    # Federated Learning Server
    # ============================================================
    fed = Fed_server(
        network, criterion, config, per_client_data, per_client_label,
        training_num_per_cls, test_data, test_label, state_list, model_list,
        eval_result_list=eval_result_list, size_per_client=size_per_client,   
        size_prob=size_prob, lr_collect=lr_collect, 
        size_accounter=size_accounter, time_record=time_record,
        drop_prob=drop_prob, round_conter=round_conter,
        waiting_time_record=waiting_time_record    
    )

    aggre_mode = config["fl_opt"]["aggregation"]
    frac = config["fl_opt"]["frac"]
    client_per_round = max(int(frac * fed.num_clients), 1)

    # ============================================================
    # Federated Learning Clients (Multi-Process)
    # ============================================================
    for i in range(num_client):
        p = Fed_client(
            network, criterion, config, per_client_data, per_client_label,
            training_num_per_cls, test_data, test_label, state_list, model_list,
            eval_result_list, idx=i, size_i=size_per_client[i],
            size_prob=size_prob, lr_collect=lr_collect, 
            size_accounter=size_accounter, time_record=time_record,
            drop_prob=drop_prob, round_conter=round_conter,
            waiting_time_record=waiting_time_record
        )
        p.daemon = True
        p.start()
        process_list.append(p)

    # ============================================================
    # Federated Learning Training Loop
    # ============================================================
    waiting_time = 0
    if not config['test']:
        best_acc = 0

        for round_i in tqdm(range(config["fl_opt"]["rounds"]), desc="FL Rounds"):   
            round_conter.value = round_i
            
            # Select clients for this round
            selected_idx = np.random.choice(
                range(fed.num_clients), client_per_round, replace=False
            )
            
            selected_cls = []
            for i in selected_idx:
                selected_cls += list(cls_per_client[i])
            print_write([f"\n\nRound: {round_i}, Selected clients: {selected_idx}"], log_file)
            
            # Federated training steps
            fed.local_train(selected_idx)
            fed.aggregation(selected_idx, aggre_mode)
            waiting_time += max(fed.waiting_time_record) - min(fed.waiting_time_record)

            # Multi-scale update for adaptive resolution
            fed.multi_scale_update()
            
            # Evaluation
            train_loss_array, train_acc_array, test_loss_array, test_acc_array = \
                fed.evaluate_global_size_hetero(skip_train=True)
            kd_loss = np.array(fed.losses_kd)[selected_idx].mean()
            cls_loss = np.array(eval_result_list).mean()

            # ============================================================
            # Logging and Results
            # ============================================================
            np.set_printoptions(precision=3)

            if config["hetero_size"]["train_hetero"]:       
                # Per-size accuracy metrics
                train_loss_list = train_loss_array.mean(1)
                train_acc_list = train_acc_array.mean(1)
                test_loss_list = test_loss_array.mean(1)
                test_acc_list = test_acc_array.mean(1)

                num_of_client = config["fl_opt"]["num_clients"]
                train_loss_mean = np.sum(train_loss_list * number_per_size) / num_of_client
                train_acc_mean = np.sum(train_acc_list * number_per_size) / num_of_client
                test_loss_mean = np.sum(test_loss_list * number_per_size) / num_of_client
                test_acc_mean = np.sum(test_acc_list * number_per_size) / num_of_client       
                
                print_write(
                    [
                        "cls_loss, kd_loss, train_loss, test_loss, train_acc, test_acc, test_acc_mean: ",
                        [cls_loss, kd_loss, train_loss_mean, test_loss_mean, 
                         train_acc_list, test_acc_list, test_acc_mean],
                    ],
                    log_file,
                )
            else:       
                # Per-class accuracy metrics
                train_loss_list = train_loss_array.mean(0)
                train_acc_list = train_acc_array.mean(0)
                test_loss_list = test_loss_array.mean(0)
                test_acc_list = test_acc_array.mean(0)
                
                train_loss_mean = train_loss_list.mean()
                train_acc_mean = train_acc_list.mean()
                test_loss_mean = test_loss_list.mean()
                test_acc_mean = test_acc_list.mean()
                
                print_write(["per_cls_acc (train): ", train_acc_list], log_file)
                print_write(["per_cls_acc (test): ", test_acc_list], log_file)

            # Write to tensorboard
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
                acc_16=test_acc_list[0], acc_24=test_acc_list[1], 
                acc_32=test_acc_list[2],
            )
            
            # Track best accuracy
            if test_acc_mean > best_acc:
                best_acc = test_acc_mean
                print_write(
                    [f"Best round: {round_i}, accuracy: {test_acc_mean*100:.2f}%"], 
                    log_file
                )

    # ============================================================
    # Final Statistics
    # ============================================================
    total_computation = compute_total_computation(size_accounter)
    avg_waiting_time = waiting_time / config['fl_opt']['rounds']

    print_write(["Total computation: ", total_computation], log_file)
    print_write(["Average waiting time per round: ", avg_waiting_time], log_file)

    # Cleanup processes
    for p in process_list:
        p.terminate()

