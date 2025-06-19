import os
import torch
import argparse

from util import set_seed
from configure import get_default_config
from runmodule import RunModule

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset = {
    0: "HandWritten",  # this
    1: "aloideep3v",
    2: "Caltech101-7",
    3: "Fashion",
    4: "Scene-15",
    5: "CIFAR10",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='20', help='number of test times')
parser.add_argument('--temperature_f', type=float, default=0.5, help='temperature parameter')
parser.add_argument('--feature_dim', default=512)
parser.add_argument('--high_feature_dim', default=128)

args = parser.parse_args()
dataset = dataset[args.dataset]

def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['Dataset']['name'] = dataset
    config['temperature_f'] = args.temperature_f
    config['feature_dim'] = args.feature_dim
    config['high_feature_dim'] = args.high_feature_dim

    # set seed
    set_seed(config['training']['seed'])

    # training module
    run = RunModule(config, device)

    for run.cfg['training']['loss_weight1'] in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
        for run.cfg['training']['lambda_graph'] in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
            print(f"Current lambda_CL: {run.cfg['training']['loss_weight1']}, lambda_graph: {run.cfg['training']['loss_weight2']}")
            
            run.recover_train()
            run.contrastive_train()
            run.SelfPaced_train()
    


if __name__ == '__main__':
    main()