"""Entry point."""


import time
import torch
import argparse
import numpy as np
from sys import exit
import graphnas.utils.tensor_utils as utils
from graphnas.evolution_trainer import Evolution_Trainer


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS,\
                              derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--max_save_num', type=int, default=5)
    # EA
    parser.add_argument('--cycles', type=int, default=1000,
                        help='Evolution cycles')
    parser.add_argument('--eval_cycle', type=int, default=100,
                        help='Evaluate best model every x iterations. def:100')
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=25,
                        help='Sample size for tournament selection')
    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="Citeseer",
                        required=False, help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file',
                        type=str,
                        default=f"sub_manager_logger_file_{time.time()}.txt")


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.set_printoptions(precision=8)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trainer = Evolution_Trainer(args)
    if args.mode == 'train':
        print(args)
        trainer.train()
    elif args.mode == 'derive':
        trainer.derive()
    else:
        raise Exception("[!] Mode not found: ", args.mode)


if __name__ == "__main__":
    args = build_args()
    main(args)
