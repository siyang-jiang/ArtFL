from cmath import cos
import copy
from distutils.command.config import config
from random import shuffle
from re import L
import threading
import time
from collections import OrderedDict

import torch
from data.dataloader import local_client_dataset, test_dataset
from models.utils import *
from utils.train_helper import validate_one_model
from utils.sampling import *
from utils.misc import check_nan

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from multiprocessing import Process
import torchvision
import time


def return_state_dict(network):
    """
    save model to state_dict
    """
    feat_model = {k: v.cpu() for k, v in network["feat_model"].state_dict().items()}
    classifier = {k: v.cpu() for k, v in network["classifier"].state_dict().items()}
    return {"feat_model": feat_model, "classifier": classifier}


def load_state_dict(network, state_dict):
    """
    restore model from state_dict
    """
    network["feat_model"].load_state_dict(state_dict["feat_model"])
    network["classifier"].load_state_dict(state_dict["classifier"])

    return network


def check_status(status_list, selected_idx, target_status):
    """
    Base status: 
    ---
    0. original status (1st FL round)
    1. server finished sending: server_network --> mp_list
    2. client received, updated and returned the model: mp_list --> networks[i] --> local_update --> mp_list
    Aftert the aggregation finished: networks[i] --> aggregate --> server_network --> mp_list, the status change to 1 
        
    Additional status for personalized FL: 
    ---
    3. server finished sending: server_network --> mp_list. But it is in meta test stage where the local_ep==1 (Per-FedAvg algorithm)

    Return 
    ---
    True: when all clients meet conditions, else False
    """
    tmp = np.array(status_list)
    if (tmp[selected_idx] == target_status).all() == True:
        return True
    else:
        return False


def set_status(status_list, selected_idx, target_status):
    """
    see function: check_status
    """
    if type(selected_idx) is int:
        selected_idx = [selected_idx]
    for i in selected_idx:
        status_list[i] = target_status

def difference_models_norm_2(model_1, model_2):
    """
    Return the norm 2 difference between the two model parameters. Used in FedProx, FedDyn
    """
    tensor_1_backbone = list(model_1["feat_model"].parameters())
    tensor_1_classifier = list(model_1["classifier"].parameters())
    tensor_2_backbone = list(model_2["feat_model"].parameters())
    tensor_2_classifier = list(model_2["classifier"].parameters())
    
    diff_list = [torch.sum((tensor_1_backbone[i] - tensor_2_backbone[i])**2) for i in range(len(tensor_1_backbone))]
    diff_list.extend([torch.sum((tensor_1_classifier[i] - tensor_2_classifier[i])**2) for i in range(len(tensor_1_classifier))])

    norm = sum(diff_list)
    return norm


import torch.nn.functional as F
import torch.nn as nn
from models.model import BasicBlock, _weights_init

class personalized_network_16(nn.Module):
    """
    return personalized backbone
    """
    def __init__(self, backbone):
        super(personalized_network_16, self).__init__()
        # resize_head
        self.conv1_16 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_16 = nn.BatchNorm2d(128)
        self.relu1_16 = nn.ReLU()
        """Using exec() will cause bugs here
        exec(f"self.layer1_{self.size} = BasicBlock(64, 128, stride=1, downsample=None, option='B')")
        """

        # shared layers from backbone
        self.layer3 = backbone.layer3
        self.cb_block = backbone.cb_block
        self.rb_block = backbone.rb_block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # init weight
        self.apply(_weights_init)

    def forward(self, x):
        # output of resize_head (n,128,16,16)
        x = self.relu1_16(self.bn1_16(self.conv1_16(x)))
        """
        Using exec() will cause bugs here
        exec(f"x = self.layer1_{self.size}(x)")
        """

        # shared layers 
        x = self.layer3(x)
        out1 = self.cb_block(x)
        out2 = self.rb_block(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        return out


class personalized_network_24(nn.Module):
    """
    return personalized backbone
    """
    def __init__(self, backbone):
        super(personalized_network_24, self).__init__()
        self.size = 24
        # resize_head
        self.conv1_24 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_24 = nn.BatchNorm2d(128)
        self.relu1_24 = nn.ReLU()

        # shared layers from backbone
        self.layer3 = backbone.layer3
        self.cb_block = backbone.cb_block
        self.rb_block = backbone.rb_block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # init weight
        self.apply(_weights_init)

    def forward(self, x):
        # output of resize_head (n,128,16,16)
        x = self.relu1_24(self.bn1_24(self.conv1_24(x)))
        x = F.interpolate(x, size=(16, 16))
    
        # shared layers 
        x = self.layer3(x)
        out1 = self.cb_block(x)
        out2 = self.rb_block(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        return out



class Fed_server(Process):
    """
    Class for client updating and model aggregation
    """
    def __init__(
        self, network, criterion, config, per_client_data, 
        per_client_label, training_num_per_cls,
        test_data, test_label, 
        state_list=None, state_dict_list=None, 
        eval_result_list=None, idx=None, 
        size_per_client=None, size_prob=None, 
        lr_collect=None, size_accounter = None, time_record = None, drop_prob = None, round_conter = None, waiting_time_record = None
        ):

        super(Fed_server, self).__init__()

        self.local_bs = config["fl_opt"]["local_bs"]
        self.local_ep = config["fl_opt"]["local_ep"]
        self.num_clients = config["fl_opt"]["num_clients"]
        self.criterion = criterion
        self.networks, self.optimizers, self.optimizers_stage2, self.schedulers = [], [], [], []
        self.train_loaders = []     # include dataloader or pre-loaded dataset
        self.local_num_per_cls = []   # list to store local data number per class
        self.test_loaders = []
        self.status_list = state_list
        self.state_dict_list = state_dict_list
        self.eval_result_list = eval_result_list
        self.client_idx = idx   # physical idx of clients (hardcoded)
        self.lr_collect = lr_collect
        self.size_accounter = size_accounter
        self.size_per_client = size_per_client
        self.sizes = config["hetero_size"]["sizes"] 

        self.prob = [[0.6, 0.2, 0.2], [1/3, 1/3, 1/3], [0.2, 0.2, 0.6]]
        self.size_prob = size_prob
        self.size_prob_previous = [ [1/3, 1/3, 1/3] for i in range(self.num_clients)]
        
        self.round_counter = 0
        self.time_record = time_record
        self.waiting_time_record = waiting_time_record

        self.drop_prob = drop_prob
        self.config = config
        self.feat_aug = config["fl_opt"]["feat_aug"]
        self.crt = config["fl_opt"]["crt"]

        self.client_weights = np.array([i for i in training_num_per_cls])
        self.client_weights = self.client_weights/self.client_weights.sum()

        # per-client accuracy and loss
        self.acc = [0 for i in range(self.num_clients)]
        self.losses_cls = [-1 for i in range(self.num_clients)]
        self.losses_kd = [-1 for i in range(self.num_clients)]

        ######## init backbone, classifier, optimizer and dataloader ########
        network = init_models(config) 

        self.device = torch.device(config['GPU'] if torch.cuda.is_available() else 'cpu')
        self.server_network = copy.deepcopy(network)    # network for original size
        self.server_network["feat_model"].to(self.device)
        self.server_network["classifier"].to(self.device)

        # feddyn
        if self.config["fl_opt"]["aggregation"] == "feddyn":
            self.h = {
                "feat_model": dict([ [key, torch.zeros_like(values)] for key, values in self.server_network["feat_model"].state_dict().items() ]), 
                "classifier": dict([ [key, torch.zeros_like(values)] for key, values in self.server_network["classifier"].state_dict().items() ]), 
            } 
            self.alpha = 0.005

        # server also stores all clients' network
        for client_i in range(self.num_clients):
            backbone = copy.deepcopy(network["feat_model"])   
            if config["fl_opt"]["branch"] is True:    
                if size_per_client[client_i] == 16:
                    backbone = personalized_network_16(backbone)
                elif size_per_client[client_i] == 24:
                    backbone = personalized_network_24(backbone)
            classifier = copy.deepcopy(network["classifier"])
            self.networks.append(
                {"feat_model": backbone.to(self.device), "classifier": classifier.to(self.device)})

        # Find layer-client relationship. Establish global state_dict for all layers
        self.layer_client_table = {}
        self.state_dict_all = {}
        for client_i in range(self.num_clients):
            for net in self.networks[client_i].values():  # feat_model / classifier
                for key, layer in net.state_dict().items():
                    if key not in self.layer_client_table:
                        self.layer_client_table.update({key: []})  
                        self.state_dict_all.update({
                            key: torch.zeros_like(layer)
                            })
                    self.layer_client_table[key].append(client_i)


        self.round_conter = round_conter
        self.dataset_name = self.config['dataset']['name']

        train_data_all, train_label_all = [], []
        for client_i in range(len(per_client_label)):
            train_data_all = train_data_all + per_client_data[client_i]
            train_label_all = train_label_all + per_client_label[client_i]


        self.train_dataset = local_client_dataset(train_data_all,
                                                  train_label_all,
                                                  config,
                                                  phase="test",
                                                  client_id=None
                                                  ) # test training set on test resolution
        self.test_dataset = local_client_dataset(test_data,
                                                 test_label,
                                                 config,
                                                 phase="test",
                                                 client_id=None)

        self.test_data = test_data
        self.test_label = test_label

    def local_train(self, selected_idx, meta_test=False):
        """
        server-side code
        """
        # model transfer: self.networks --> mp_list
        for i in selected_idx:
            self.state_dict_list[i] = return_state_dict(self.networks[i])  
        set_status(self.status_list, selected_idx, 1)  

        # wait until all clients returning the model
        while check_status(self.status_list, selected_idx, 2) is False:
            time.sleep(0.1)

        # mp_list --> self.networks (copys of client models on the server). Prepare for aggregation.
        for i in selected_idx:
            load_state_dict(self.networks[i], self.state_dict_list[i])  # model transfer
    
        print("===> Local training finished")


    def aggregation(self, selected_idx, mode):
        """
        server-side code: aggregation
        """
        if mode in ["fedavg", "fedbn", "fedprox", "feddyn"]:
            self.aggregate_layers(selected_idx, mode)
            # self.aggregate_layers_adapt(selected_idx, mode)
        else:
            raise RuntimeError

        # model: self.networks --> mp_list
        for i in selected_idx:
            self.state_dict_list[i] = return_state_dict(self.networks[i])  # model transfer
        set_status(self.status_list, selected_idx, 0)    # back to original 
        
        print("===> Aggregation finished")


    def drop_update(self):
        '''
        Update drop scheme: Drop x% data in each client
        '''
        return self.drop_scheme()


    def drop_scheme(self):
        '''
        time_record: Recived time_record is the revised by cost function
        '''
     
        cost_list = [1, 2.25, 4]

        for i in range(len(self.size_prob)):
            cost = 0
            for j in range(len(cost_list)):
                cost += cost_list[j] * self.size_prob[i][j] 
    
            tmp1 = self.time_record[i] 
            self.time_record[i] = self.time_record[i] * cost
            tmp2 = self.time_record[i]
    
            if tmp1 - tmp2 == 0 and cost[i] != 1:
                raise ValueError("No Change time record")
        
        x_mean =  np.mean(self.time_record)
        x_var = np.var(self.time_record)  

        #### End To Compute Part 2 of Time ##########

        drop_prob_list = [0.4, 0.2, 0.1]
        for i in range(len(self.time_record)):        
            if self.time_record[i] > (x_mean):
                self.drop_prob[i] = ( self.time_record[i] -  x_mean) / self.time_record[i]
            else:
                self.drop_prob[i] = drop_prob_list[2]
        return self.drop_prob 


    def multi_scale_update(self):
        """
        server-side code: determine the size for each clients
        """
        self.size_schedule()
        if self.config["random_drop"]:
            self.drop_update()
        return 1


    def size_schedule(self):
        '''
        Size_schedule baseline
        Using loss and time to determine the ranking.
        - time_loss: a.k.a ranking score
        '''
        # ranking

        # standard setting
        if self.config['fl_opt']['num_clients'] == 10:
            group = [2,2,2, 1,1,1,1, 0,0,0]   

        elif self.config['fl_opt']['num_clients'] == 26:
            
            group = [2,2,2,2, 2,2,2,2, 2,2,2,2,
            1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,
            0,0,0,0, 0,0,0,0, 0,0,0,0]

        elif self.config['fl_opt']['num_clients'] == 30:
            group = [2,2,2,2,2, 2,2,2,2,2,
            1,1,1,1,1, 1,1,1,1,1,
            0,0,0,0,0, 0,0,0,0,0]
        
        elif self.config['fl_opt']['num_clients'] == 50:
            group = [2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,
            1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,
            0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0]
        
        elif self.config['fl_opt']['num_clients'] == 100:
            group = [
            2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 
            2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2,
            2,2,2,
            1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 
            1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 
            1,1,1,1,
            0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
            0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
            0,0,0
            ]
        # centrailized training
        elif self.config['fl_opt']['num_clients'] < 10:
            group = [2 for i in range( self.config['fl_opt']['num_clients'])]

        if self.config['fl_opt']['frac'] < 1:

            time_loss = []
            for i in range(len(self.time_record)):
                if self.eval_result_list[i] is None:
                    self.eval_result_list[i] = 0
    
                tmp = self.time_record[i] * self.eval_result_list[i]
                time_loss.append(tmp)
        else:
            time_loss = [self.time_record[i] * self.eval_result_list[i] for i in range(len(self.time_record))]

        sorted_list = sorted(time_loss)

        idx = [sorted_list.index(i) for i in time_loss] # the idx-th smallest
        
        print("Time record=====>" + str( self.time_record) )
        for i in idx:
            tmp = self.size_prob[i]
            tmp = self.prob[group[i]] # all [1/3 1/3 1/3], shuai baseline
            self.size_prob[i] = tmp

        return self.size_prob

    def aggregate_layers_adapt(self, selected_idx, mode):
        """
        self-adaptive aggregation for hetergeneous models based on layer names
        """
        with torch.no_grad(): 

            # update the state_dict_all that contains all layers 
            for key, client_list in self.layer_client_table.items():
                
                # clients that have this layer (key) AND are selected                
                new_selected_idx = list(set(selected_idx).intersection(set(client_list)))
                weights_sum = self.client_weights[new_selected_idx].sum() 

                if 'num_batches_tracked' in key:
                    self.state_dict_all[key] = self.networks[new_selected_idx[0]]["feat_model"].state_dict()[key]
                    
                else:       
                    # Fedavg/FedProx/FedBN
                    temp = torch.zeros_like(self.state_dict_all[key])
                    for client_i in new_selected_idx:
                        if key in ["fc.weight", "weight_scale"]:    
                            layer = self.networks[client_i]["classifier"].state_dict()[key]
                        else:
                            layer = self.networks[client_i]["feat_model"].state_dict()[key]
                        
                        weight = self.client_weights[client_i] / weights_sum
                        temp += weight * layer

                    self.state_dict_all[key] = temp

            # update self.networks according to self.state_dict_all
            for client_i in range(self.num_clients):    
                for net_name in ["feat_model", "classifier"]:
                    for key, layer in self.networks[client_i][net_name].state_dict().items():
                        if 'bn' in key and mode == "fedbn":
                            pass
                        else:
                            layer.data.copy_(self.state_dict_all[key])


    def aggregate_layers(self, selected_idx, mode):
        """
        Need the the network structure on each client to be the same. 
        fedavg, fedprox, fedbn, feddyn
        """
        num_clients_tmp = self.config['fl_opt']['num_clients']
        frac = self.config['fl_opt']['frac_cp']

        if frac < 1:
            selected_idx_tmp = selected_idx
            length = int(frac * num_clients_tmp)
            tmp_arr = np.arange(num_clients_tmp)
            np.random.shuffle(tmp_arr)
            selected_idx = tmp_arr[:length]
            
            print('actual select idx' + str(selected_idx) )

        weights_sum = self.client_weights[selected_idx].sum()
        with torch.no_grad():
            if mode in ["fedavg", "fedprox", "fedbn"]:

                    
                for net_name, net in self.server_network.items():
                    for key, layer in net.state_dict().items():
                        if 'num_batches_tracked' in key:
                            layer.data.copy_(self.networks[0][net_name].state_dict()[key])
                        else: 
                            temp = torch.zeros_like(layer)
                            for idx in selected_idx:
                                weight = self.client_weights[idx]/weights_sum
                                temp += weight * self.networks[idx][net_name].state_dict()[key]
                            layer.data.copy_(temp) 

            if mode == "feddyn":
                for net_name, net in self.server_network.items():
                    for key, layer in net.state_dict().items():
                        if 'num_batches_tracked' in key:
                            layer.data.copy_(self.networks[0][net_name].state_dict()[key])
                        else: 
                            temp = torch.zeros_like(layer)
                            for idx in selected_idx:
                                weight = self.client_weights[idx]/weights_sum
                                temp += weight * self.networks[idx][net_name].state_dict()[key]

                            self.h[net_name][key] -= self.alpha * \
                                (temp - self.server_network[net_name].state_dict()[key])

                            temp -= (1./self.alpha) * self.h[net_name][key]
                            layer.data.copy_(temp) 
           

            for idx in range(self.num_clients):
                self.networks[idx] = copy.deepcopy(self.server_network)
        
        if frac < 1:
            selected_idx = selected_idx_tmp

    def change_bn_status(self, tracking_stat):
        """
        FedBN: testing on unknown dataset, in line with SiloBN.
        """
        for idx in range(self.num_clients):
            for name, layer in self.networks[idx]["feat_model"].named_modules():
                if "bn" in name:
                    layer.track_running_stats = tracking_stat                  


    def evaluate_global(self, train_dataset=None, test_dataset=None):
        """
        One global model.
        For training set, return the mean loss/acc of all classes.
        For test set, return the mean loss/acc according to shot numbers.
        """
        # evaluate on training set
        if train_dataset is None:
            train_dataset = self.train_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset

        train_loss_per_cls, train_acc_per_cls = validate_one_model(
            self.server_network, train_dataset, self.device, per_cls_acc=True) 

        # evaluate on test set: per-class loss/acc
        test_loss_per_cls, test_acc_per_cls = validate_one_model(
            self.server_network, test_dataset, self.device, per_cls_acc=True) 
        print("===> Evaluation finished\n")

        return train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls


    def evaluate_global_size_hetero(self, train_dataset=None, test_dataset=None, skip_train=False):
        """
        Multiple global model
        Accuracy of model under every size of every class.
        If fast is True: skip training set 
        ---
        Return:
        all_results: shape (4, num_size, num_cls), 4 for (train_loss, train_acc, test_loss, test_acc)
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset
        num_cls = self.config["dataset"]["num_classes"]
        
        if self.config["fl_opt"]["aggregation"] == "fedbn":
            self.change_bn_status(tracking_stat = True)

        all_results = [None for i in range(len(self.sizes))]
        # reduce three times
        if self.dataset_name in ['Speech'] and self.config['use_resize_data']:            
            for size_idx, size in enumerate(self.sizes):

                if size not in self.size_per_client:
                    all_results[size_idx] = [np.zeros(num_cls)]*4
                    continue

                client_idx = list(self.size_per_client).index(size)

                if skip_train is True:
                    train_loss_per_cls, train_acc_per_cls = np.zeros(num_cls), np.zeros(num_cls)
                else:
                    train_loss_per_cls, train_acc_per_cls = validate_one_model(
                        self.networks[client_idx], train_dataset, self.device, per_cls_acc=True)
                # evaluate on test set: per-class loss/acc
                max_length = int(len(test_dataset.label) / 3)
                tmp = 2 * max_length
                if size == 32:
                    test_data_tmp = self.test_data[:max_length]
                    test_label_tmp = self.test_label[:max_length]

                elif size == 24:

                    test_data_tmp = self.test_data[max_length: tmp]
                    test_label_tmp = self.test_label[max_length: tmp]

                elif size == 16: 
                    test_data_tmp = self.test_data[tmp:]
                    test_label_tmp = self.test_label[tmp:]

                else:
                    raise ValueError("Bad Size")


                test_dataset_eval = local_client_dataset(test_data_tmp,
                                                  test_label_tmp,
                                                  self.config,
                                                  phase="test",
                                                  client_id=None
                                                  ) # test training set on test resolution

                test_loss_per_cls, test_acc_per_cls = validate_one_model(
                    self.networks[client_idx], test_dataset_eval, self.device, per_cls_acc=True) 

                all_results[size_idx] = train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls
            
                print(f"===> Evaluation finished{size_idx}")        
    
        else:
            for size_idx, size in enumerate(self.sizes):

                if self.dataset_name in ['CIFAR10', 'CIFAR100']:
                    if self.config["fl_opt"]["num_clients"] == 1:
                        self.size_per_client = [16, 24, 32]
                
                    elif self.config["motivation_flag"]:
                        self.size_per_client = [16, 24, 32]
                    
                    elif self.config["update_eval"]:
                        self.size_per_client = [16, 24, 32]
                
                if self.dataset_name in ['tiny'] and self.config["fl_opt"]["num_clients"] == 1:
                    self.size_per_client = [32, 48, 64]

                if size not in self.size_per_client:
                    all_results[size_idx] = [np.zeros(num_cls)]*4
                    continue
                
                if self.dataset_name in ['CIFAR10', 'CIFAR100', 'tiny'] and self.config["fl_opt"]["num_clients"] < 3:
                    client_idx = 0
                else:
                    client_idx = list(self.size_per_client).index(size)

                train_dataset.eval_size_id = size_idx
                test_dataset.eval_size_id = size_idx
                train_dataset.update_transform()               
                test_dataset.update_transform()

                # evaluate on training set: per-class loss/acc
                if skip_train is True:
                    train_loss_per_cls, train_acc_per_cls = np.zeros(num_cls), np.zeros(num_cls)
                else:
                    train_loss_per_cls, train_acc_per_cls = validate_one_model(
                        self.networks[client_idx], train_dataset, self.device, per_cls_acc=True)
                # evaluate on test set: per-class loss/acc
                test_loss_per_cls, test_acc_per_cls = validate_one_model(
                    self.networks[client_idx], test_dataset, self.device, per_cls_acc=True) 
                    
                all_results[size_idx] = train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls
                print(f"===> Evaluation finished{size_idx}")        
        
        all_results = np.array(all_results).transpose(1,0,2)  

        if self.config["fl_opt"]["aggregation"] == "fedbn":
            self.change_bn_status(tracking_stat = True)

        return all_results

    def get_size_accounter(self):
        return self.size_accounter


class Fed_client(Process):
    """
    Class for client updating and model aggregation
    """
    def __init__(
        self, network, criterion, config, 
        per_client_data, per_client_label, training_num_per_cls, test_data, test_label, 
        state_list=None, state_dict_list=None, eval_result_list=None, idx=None, size_i=None, size_prob=None, 
        lr_collect=None, size_accounter = None, time_record = None, drop_prob = None, round_conter = None, waiting_time_record = None

        ):

        super(Fed_client, self).__init__()

        self.local_bs = config["fl_opt"]["local_bs"]
        self.local_ep = config["fl_opt"]["local_ep"]
        self.num_clients = config["fl_opt"]["num_clients"]
        self.criterion = criterion
        self.networks, self.optimizers, self.optimizers_stage2, self.schedulers = [], [], [], []
        self.train_loaders = []     # include dataloader or pre-loaded dataset
        self.train_loader_balanced = []     
        self.local_num_per_cls = []   # list to store local data number per class
        self.test_loaders = []
        self.status_list = state_list
        self.state_dict_list = state_dict_list
        self.eval_result_list = eval_result_list
        self.client_idx = idx   # physical idx of clients (hardcoded)
        self.config = config
        self.sizes = config["hetero_size"]["sizes"]
        
        self.feat_aug = config["fl_opt"]["feat_aug"]
        self.crt = config["fl_opt"]["crt"]

        self.size_prob = size_prob
        self.size_prob_tmp = 0
        self.lr_collect = lr_collect
        self.size_accounter = size_accounter
        self.time_record = time_record
        self.drop_prob = drop_prob
        self.changed_size = 32
        self.mu = 0.1     # fedprox  

        self.round_conter = round_conter
        self.waiting_time_record = waiting_time_record

        # federated aggregation weight
        self.client_weights = np.array([i for i in training_num_per_cls])
        self.client_weights = self.client_weights/self.client_weights.sum()

        # per-client accuracy and loss
        self.acc = [0 for i in range(self.num_clients)]
        self.losses_cls = [-1 for i in range(self.num_clients)]
        self.losses_kd = [-1 for i in range(self.num_clients)]
        print(f'=====> {config["metainfo"]["optimizer"]}, Client {idx} (fed.py)\n ')        


        ######## init backbone, classifier, optimizer ########
        self.device = config["device_client"][idx]
        for client_i in range(self.num_clients):
            # list of network and optimizer_dict. One optimizer for one network.
            if client_i != self.client_idx:
                self.networks.append(None)
                self.optimizers.append(None)
            else: 
                backbone = copy.deepcopy(network["feat_model"])       
                classifier = copy.deepcopy(network["classifier"])
                if config["fl_opt"]["branch"] is True:
                    if size_i == 16:
                        backbone = personalized_network_16(backbone)
                    elif size_i == 24:
                        backbone = personalized_network_24(backbone)
                self.networks.append({"feat_model": backbone, "classifier": classifier})
                self.optimizers.append(init_optimizers(self.networks[client_i], config))   

        # feddyn
        if config["fl_opt"]["aggregation"] == "feddyn":
            self.alpha = 0.1
            self.prev_grads = None
            for net_name, net in self.networks[self.client_idx].items():
                for param in net.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = torch.zeros_like(param.view(-1))
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)


        ######## init per-client data ########
        num_workers = 0
        self.local_dataset = local_client_dataset(
            per_client_data[self.client_idx], 
            per_client_label[self.client_idx], 
            config, 
            "train", 
            self.client_idx, 
            size_align=config["fl_opt"]["size_align"],
            size_prob= self.size_prob,
            changed_size = self.changed_size
            )
        if config["fl_opt"]["multi_scale"]:
            collate_fn = self.local_dataset.collate_fn
            if config['dataset']['name'] == 'Speech' and config['use_resize_data']:
                collate_fn = None
        else:
            collate_fn = None

        self.train_loader = torch.utils.data.DataLoader(
            self.local_dataset, batch_size=self.local_bs, shuffle=True, 
            num_workers=num_workers, pin_memory=False, 
            collate_fn=collate_fn
            )
        self.train_loader_balanced = torch.utils.data.DataLoader(
            self.local_dataset, batch_size=self.local_bs, 
            sampler=self.local_dataset.get_balanced_sampler(), 
            num_workers=num_workers, pin_memory=True
            )
        self.per_client_data = per_client_data
        self.per_client_label = per_client_label
        

    def run(self):
        """
        client-side code
        """
        self.networks[self.client_idx]["feat_model"].to(self.device)
        self.networks[self.client_idx]["classifier"].to(self.device)

        while(1):   
            while check_status(self.status_list, self.client_idx, 1) is False:
                time.sleep(0.01)
        
            load_state_dict(self.networks[self.client_idx], self.state_dict_list[self.client_idx])

            self.local_dataset.size_prob = self.size_prob[self.client_idx]

            ## Just for motivation experiments
            if self.config['motivation_flag']:
                            
                if 0 <= self.round_conter.value < 200:
                    self.changed_size = 32
                elif 200 <= self.round_conter.value < 400 :
                    self.changed_size = 24
                elif 400 <= self.round_conter.value < 600 :
                    self.changed_size = 16

                self.changed_size = 8
                self.local_dataset.changed_size = self.changed_size

            if self.config["fl_opt"]["multi_scale"]:
                collate_fn = self.local_dataset.collate_fn
                self.train_loader = torch.utils.data.DataLoader(
                    self.local_dataset, batch_size=self.local_bs, shuffle=True, 
                    num_workers=0, pin_memory=False, collate_fn=collate_fn
                )

            start = time.time()
            self.train_lt(self.client_idx)    
            end = time.time()
            
            self.waiting_time_record[self.client_idx] = end - start
            revised_time = self.estimate_computing_time(end - start)
            self.time_record[self.client_idx] = revised_time #Note that we only compute a part of time_record

            self.state_dict_list[self.client_idx] = return_state_dict(self.networks[self.client_idx])      
            set_status(self.status_list, self.client_idx, 2)


    def estimate_computing_time(self, time):
        '''
        T(i,r+1) = C(i,r+1) * T(i,r)/ C(i,r)   # predict the new delay for each client 
        '''
        size_prob_tmp = self.size_prob[self.client_idx] #Get the prob in previous round
        cost_list = [1, 2.25, 4]
        cost = np.sum([cost_list[i] * size_prob_tmp[i] for i in range(len(cost_list))])

        revised_time_1 = time / cost / (1-self.drop_prob[self.client_idx]) 
        return revised_time_1


    def evaluate_global(self, train_dataset=None, test_dataset=None):
        """
        For training set, return the mean loss/acc of the all classes.
        For test set, return the mean loss/acc according to shot numbers.
        """
        # evaluate on training set
        if train_dataset is None:
            train_dataset = self.train_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset

        train_loss_per_cls, train_acc_per_cls = validate_one_model(
            self.networks[self.client_idx], train_dataset, self.device, per_cls_acc=True) 

        # evaluate on test set: per-class loss/acc
        test_loss_per_cls, test_acc_per_cls = validate_one_model(
            self.networks[self.client_idx], test_dataset, self.device, per_cls_acc=True) 
        print(f"===> Evaluation finished{self.client_idx}")
        
        return (train_loss_per_cls, train_acc_per_cls, test_loss_per_cls, test_acc_per_cls)
        

    def optimizer_step(self, idx_in_all):
        for optimizer in self.optimizers[idx_in_all].values():
            optimizer.step()

    def optimizer_zero_grad(self, idx_in_all):
        for optimizer in self.optimizers[idx_in_all].values():
            optimizer.zero_grad()


    def train_lt(self, idx):   
        """
        client-side code
        ---
        Argus:
        - idx: the index in all clients (e.g., 50)
        """ 
        for net in self.networks[idx].values():
            net.train()
        teacher = copy.deepcopy(self.networks[idx])
        for net in teacher.values():
            net.train()  

        # torch.cuda.empty_cache()

        with torch.set_grad_enabled(True):  
            losses_cls = 0

            # print(self.drop_prob[idx])
            for epoch in range(self.local_ep):   
                # flag = 0
                # if epoch == 0:
                #     flag = 1

                if self.config["fl_opt"]["balanced_loader"]:
                    tmp_loader = self.train_loader_balanced
                else:
                    tmp_loader = self.train_loader

                tmp = random.randint(1,10) / 10

                if self.config["random_drop"] and tmp <= self.drop_prob[idx]:
                    continue


                # Make a copy to random drop x% 「Batch Level (Need to test)」
                # max_dataset_len = len(self.train_loader) * (1 - self.drop_prob[idx])

                inner_data_counter = 0
                for (imgs, labels, indexs) in tmp_loader:
                    
                    # forward
                    imgs = imgs.to(self.device)
                    feat = self.networks[idx]['feat_model'](imgs)
                    logits = self.networks[idx]['classifier'](feat)

                    imgs_cp = copy.deepcopy(imgs)
                    # print(str(idx) + '===> image shape ===> ' + str(imgs.shape)) # B C H W 
                    B , H = list(imgs_cp.cpu().detach().numpy().shape)[0], list(imgs_cp.cpu().detach().numpy().shape)[3]

                    # count size 
                    if self.config['dataset']['name'] in ['CIFAR10', 'CIFAR100', 'tiny', 'Speech', "imu", "statefarm", "depth"] :
                        tmp_size = self.size_accounter[idx]
                        if H == self.config["hetero_size"]["sizes"][0]:
                            tmp_size[0] +=1 
                        elif H == self.config["hetero_size"]["sizes"][1]:
                            tmp_size[1] +=1 
                        elif H == self.config["hetero_size"]["sizes"][2]:
                            tmp_size[2] +=1
                        elif H == 12:
                            tmp_size[2] +=1
                        else:
                            raise NotImplementedError("Size Error")
                        self.size_accounter[idx] = tmp_size
                    else:
                        raise NotImplementedError("Dataset Error")

                    # loss    
                    labels = labels.to(self.device)
                    if self.config["criterions"]["def_file"].find("KDLoss") > 0:
                        loss, loss_cls, loss_kd = self.criterion(logits, labels, feat
                        #, feat_teacher, classfier_weight=self.networks[idx]['classifier'].fc.weight
                        )
                    elif self.config["criterions"]["def_file"].find("LwF") > 0:   
                        # teacher
                        with torch.no_grad():
                            feat_teacher = teacher['feat_model'](imgs)
                            pred_teacher = teacher['classifier'](feat_teacher)  
                        if self.feat_aug is False:
                            loss, loss_cls, loss_kd = self.criterion(labels, pred_teacher, logits)
                        else:
                            raise RuntimeError
                        
                    # fedprox loss
                    if self.config["fl_opt"]["aggregation"] == "fedprox":
                        prox_loss = difference_models_norm_2(self.networks[idx], teacher)
                        loss += self.mu/2 * prox_loss
                        # print("FedProx Loss: ", prox_loss, loss)

                    # feddyn loss
                    if self.config["fl_opt"]["aggregation"] == "feddyn":
                        # prox regularization
                        prox_loss = difference_models_norm_2(self.networks[idx], teacher)     
                        loss += self.alpha/2 * prox_loss               
                        
                        # dynamic regularization
                        lin_penalty = 0.0
                        curr_params = None
                        for net_name, net in self.networks[idx].items():
                            for name, param in net.named_parameters():
                                if not isinstance(curr_params, torch.Tensor):
                                    curr_params = param.view(-1)
                                else:
                                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)
                        self.prev_grads = self.prev_grads.to(self.device)
                        lin_penalty = torch.sum(curr_params * self.prev_grads)
                        loss -= lin_penalty
                        # print(lin_penalty, self.alpha/2 * prox_loss)

                    # backward
                    for optimizer in self.optimizers[idx].values():
                        optimizer.zero_grad()
                    loss.backward()
                    for optimizer in self.optimizers[idx].values():
                        optimizer.step()
                    
                    losses_cls += loss_cls.item()
                    inner_data_counter += 1

                    
            # feddyn: update the previous gradients
            if self.config["fl_opt"]["aggregation"] == "feddyn":
                temp_grads = None
                for net_name, net in self.networks[idx].items():
                    for name, param in net.named_parameters():
                        temp = (param - teacher[net_name].state_dict()[name]).view(-1).clone().detach() * self.alpha
                        if not isinstance(temp_grads, torch.Tensor):
                            temp_grads = temp
                        else:
                            temp_grads = torch.cat((temp_grads, temp), dim=0)                                          
                self.prev_grads -= temp_grads.to(self.device)        

            self.eval_result_list[idx] = losses_cls/len(tmp_loader)/self.local_ep

            # Round Level LR update 
            lr_accumulate = 0
            lr_count = 0
            for optimizer in self.optimizers[idx].values(): #2
                for param_group in optimizer.param_groups: #1
                    lr_accumulate += param_group['lr']
                    lr_count += 1
                lr_accumulate /= lr_count
            self.lr_collect[idx] =lr_accumulate


    def get_size_prob(self, size_prob=None):
        self.size_prob_tmp = size_prob
        return self.size_prob_tmp


def fedavg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k]*1.0, len(w))
    return w_avg


# See: https://arxiv.org/abs/1909.06335
def fedavgm(new_ws, old_w, vel, args):
    """
    fedavg + momentum
    - new_ws (list of OrderedDict): The new calculated global model
    - old_w (OrderedDict) : Initial state of the global model (which needs to be updated here)  
    """
    global_lr = 1
    beta1 = 0
    new_w = fedavg(new_ws)

    # For the first round: initialize old_w, create an Orderdict to store velocity
    if old_w is None:
        old_w = new_w
        new_v = OrderedDict()
        for key in old_w.keys():
            new_v[key] = torch.zeros(old_w[key].shape, dtype=old_w[key].dtype).to(args.device)
    else:
        new_v = copy.deepcopy(vel)

    for key in new_w.keys():
        delta_w_tmp = old_w[key] - new_w[key]
        new_v[key] = beta1*new_v[key] + torch.mul(delta_w_tmp, global_lr)
        old_w[key] -= new_v[key]

    return old_w, new_v


def fedavgw(new_ws, old_w, args, round_i):
    """
    fedavg + adaptive updating parameter
    - new_ws (list of OrderedDict): The new calculated global model
    - old_w (OrderedDict) : Initial state of the global model (which needs to be updated here)  
    """
    
    new_w = fedavg(new_ws)

    # For the first round: initialize old_w
    if old_w is None:
        old_w = new_w

    for key in new_w.keys():
        old_w[key] = new_w[key]*(1/(round_i+1)) +  old_w[key]*(round_i/(round_i+1))

    return old_w
