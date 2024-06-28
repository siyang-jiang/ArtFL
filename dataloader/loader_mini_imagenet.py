from dataset.mini_imagenet import miniImagenetFL


# Just consider the iid case
def load_mini_imagenet(train_set, eval_set, test_set, config):
    client_num = config["fl_opt"]["num_clients"]
    

    return mini_data_loader    