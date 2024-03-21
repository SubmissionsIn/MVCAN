import os
import argparse
import itertools
import torch
import random
from model import MvCAN
from util import get_logger
from datasets import *
from configure import get_default_config
import warnings
warnings.filterwarnings("ignore")

dataset = {
    1: "BDGP",
    2: "NoisyBDGP",
    3: "DIGIT",
    4: "NoisyDIGIT",
    5: "COIL",
    6: "NoisyCOIL",
    7: "Amazon",
    8: "NoisyAmazon",
    9: "DHA",
    10: "RGB-D",
    11: "Caltech-6V",
    12: "YoutubeVideo",
}
# Test = False
Test = True
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='1', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    accs = []
    nmis = []
    aris = []
    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()
    print("Dataset: " + config['dataset'])
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # Load data
    X_list, Y_list = load_data(config)
    n_clusters = len(np.unique(Y_list[0]))
    print("Cluster number: " + str(n_clusters))
    view_num = len(X_list)
    print("View number: " + str(view_num))
    view_size = []
    for v in range(view_num):
        print(X_list[v].shape)
        view_size.append(X_list[v].shape[1])

    X_train = []
    for i in range(view_num):
        view_train = X_list[i]
        X_train.append(torch.from_numpy(view_train).float().to(device))

    seed = config['training']['seed']
    acc_max = 0
    for ROUND in range(1):  # 10
        print("ROUND: " + str(ROUND+1))
        # Build the model
        Models = MvCAN(config, view_num, view_size, n_clusters=n_clusters, seed=seed, data_size=X_list[0].shape[0])
        Models.to_device(device)
        optimizers = []
        for v in range(view_num):
            optimizers.append(torch.optim.Adam(
                itertools.chain(Models.autoencoders[v].parameters()),
                lr=config['training']['lr']))
        # Print the models
        # logger.info(Models.autoencoders)
        # logger.info(optimizers)
        if Test:
            np.random.seed(seed)
            random.seed(seed)
            for v in range(view_num):
                checkpoint = torch.load('./models/' + config['dataset'] + str(v+1) + 'V.pth')
                Models.autoencoders[v].load_state_dict(checkpoint)
            print("Loading models...")
            config['training']['init_epoch'] = 0
            config['training']['epoch'] = 0
            Models.train(config, X_train, Y_list, optimizers, device)
        else:
            # Training
            acc, nmi, ari = Models.train(config, X_train, Y_list, optimizers, device, ROUND)
            if acc > acc_max:
                acc_max = acc
                for v in range(view_num):
                    state = Models.autoencoders[v].state_dict()  # each view's model is decoupled for other views
                    torch.save(state, './models/' + config['dataset'] + str(v+1) + 'V.pth')
                print('Saving...')
        if Test:
            return 0
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    print(accs, np.mean(accs), np.std(accs))
    print(nmis, np.mean(nmis), np.std(nmis))
    print(aris, np.mean(aris), np.std(aris))


if __name__ == '__main__':
    main()
