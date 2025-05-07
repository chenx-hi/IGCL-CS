def pubmed_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--cluster', type=float, default=0.05)
    parser.add_argument('--batch', type=int, default=0)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--cls_epochs', type=int, default=1000)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## mlp
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--proj_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--tau', type=float, default=0.45)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--lamda', type=float, default=0.4)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)

    ## optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.000001)

    parser.add_argument('--cls_lr', type=float, default=0.0005)
    parser.add_argument('--cls_weight_decay', type=float, default=0.05)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def amazon_computer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--k_hop', type=int, default=3)
    parser.add_argument('--cluster', type=float, default=0.04)
    parser.add_argument('--batch', type=int, default=2)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cls_epochs', type=int, default=1000)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## mlp
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--proj_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--lamda', type=float, default=0.)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)

    ## optimizer
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--cls_lr', type=float, default=0.05)
    parser.add_argument('--cls_weight_decay', type=float, default=0.0001)
    #####################################
    args, _ = parser.parse_known_args()
    return args


def coauthor_cs_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--k_hop', type=int, default=1)
    parser.add_argument('--cluster', type=float, default=0.05)
    parser.add_argument('--batch', type=int, default=0)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cls_epochs', type=int, default=600)

    parser.add_argument('--rand_split_class', type=bool, default=True,
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split', type=bool, default=False)
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    parser.add_argument('--train_ratio', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_ratio', type=float, default=.1,
                        help='validation label proportion')

    ## mlp
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--proj_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--lamda', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False)

    ## optimizer
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--cls_lr', type=float, default=0.001)
    parser.add_argument('--cls_weight_decay', type=float, default=0.00001)
    #####################################
    args, _ = parser.parse_known_args()
    return args
