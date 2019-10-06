"""Entry point."""


import time
import torch
import argparse
import numpy as np
from collections import deque
import graphnas.utils.tensor_utils as utils
from graphnas.gnn_model_manager import CitationGNNManager
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_model_manager import GeoCitationManager

# import removed from the middle of the code
from graphnas.search_space import MacroSearchSpace
from graphnas.graphnas_controller import SimpleNASController


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


def _construct_action(actions, action_list, search_space):
    structure_list = []
    for single_action in actions:
        structure = []
        # print('single_action: ', single_action)
        for action, action_name in zip(single_action, action_list):
            predicted_actions = search_space[action_name][action]
            structure.append(predicted_actions)
        structure_list.append(structure)
    return structure_list


def generate_random_individual(search_space, layers_of_child_model):
    ind = []
    for i in range(layers_of_child_model):
        ind += [np.random.randint(0, len(space))
                for key, space in search_space.items()]
    return ind


def mutate_individual(indiv, layers_of_child_model, search_space, action_list):
    # Choose a random position on the individual to mutate
    position_to_mutate = np.random.randint(len(indiv))
    # This position will receive a randomly chosen index
    # of the search_spaces's list
    # for the action corresponding to that position in the individual
    indiv[position_to_mutate] = \
        np.random.randint(0,
                          len(search_space[action_list[position_to_mutate]]))
    return indiv


def get_best_individual_accuracy(accs):
    max_acc_index = 0
    max_acc = -1
    for index, acc in enumerate(accs):
        if acc > max_acc:
            max_acc = acc
            max_acc_index = index
    return max_acc_index, max_acc


def derive_from_population(population, accs, submodel_manager):
    best_score_index, best_score = get_best_individual_accuracy(accs)
    best_structure = population[best_score_index]
    print("[DERIVE] Best Structure:" + str(best_structure))
    # train from scratch to get the final score
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    test_scores_list = []
    for i in range(10):  # run 100 times to get Mean and Stddev
        # manager.shuffle_data()
        val_acc, test_acc = submodel_manager.evaluate(best_structure)
        test_scores_list.append(test_acc)
    print(f"[DERIVE] Best Results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    search_space_cls = MacroSearchSpace()
    search_space = search_space_cls.get_search_space()
    action_list = \
        search_space_cls.generate_action_list(args.layers_of_child_model)
    print("Search space:")
    print(search_space)
    print("Generated Action List: ")
    print(action_list)
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        # implements based on dgl
        submodel_manager = CitationGNNManager(args)
    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        # implements based on pyg
        submodel_manager = GeoCitationManager(args)
    # deque with individuals (indexes to search space on action list)
    population = deque()
    # deque with the accuracies for each of the individuals in order
    accs = deque()
    cycles = args.cycles

    population_size = args.population_size
    sample_size = args.sample_size

    # Initialize population with random architectures
    print("\n\n===== Evaluating initial random population =====")
    while len(population) < population_size:
        # print('adding individual #:', len(population))
        individual = generate_random_individual(search_space,
                                                args.layers_of_child_model)
        _, ind_acc = \
            submodel_manager.train(_construct_action([individual],
                                                     action_list,
                                                     search_space)[0],
                                   format=args.format)
        print(f"individual: {individual}, val_score:{ind_acc}")
        accs.append(ind_acc)
        population.append(individual)
    print("===== Evaluating initial random population DONE ====")

    print("\n\n===== Evolution ====")
    while cycles > 0:
        sample = []  # list with indexes to population individuals
        sample_accs = []  # accuracies of the sampled individuals
        while len(sample) < sample_size:
            candidate = np.random.randint(0, len(population))
            sample.append(population[candidate])
            sample_accs.append(accs[candidate])

        # Get best individual on sample to serve as parent
        max_sample_acc_index, max_sample_acc = \
            get_best_individual_accuracy(sample_accs)
        parent = sample[max_sample_acc_index]
        # print('parent: ', parent)
        child = mutate_individual(parent, args.layers_of_child_model,
                                  search_space, action_list)
        # print('child: ', child)
        _, child_acc = \
            submodel_manager.train(_construct_action([child],
                                                     action_list,
                                                     search_space)[0],
                                   format=args.format)
        print('child acc: ', child_acc)
        print(f"parent: {parent}, val_score:{max_sample_acc} | child: {child}, val_score:{child_acc}")
        accs.append(child_acc)
        population.append(child)
        if cycles % args.eval_cycle == 0:
            derive_from_population(_construct_action(population,
                                                     action_list,
                                                     search_space),
                                   accs,
                                   submodel_manager)
        # Remove oldest individual (Aging/Regularized evolution)
        population.popleft()
        accs.popleft()
        cycles -= 1
    print("===== Evolution DONE ====")


if __name__ == "__main__":
    args = build_args()
    main(args)
