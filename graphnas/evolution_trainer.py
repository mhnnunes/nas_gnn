import time
import torch
import numpy as np
from collections import deque
from graphnas.trainer import Trainer


class Evolution_Trainer(Trainer):
    """
    This class implements the Asyncronous Aging Evolution,
    proposed by Real et. al. on:

    Regularized Evolution for Image Classifier Architecture Search

    available on: https://arxiv.org/abs/1802.01548
    """
    def __init__(self, args):
        super(Evolution_Trainer, self).__init__(args)
        self.args = args
        self.random_seed = args.random_seed
        self.population = deque()
        self.accuracies = deque()
        self.population_size = args.population_size
        self.sample_size = args.sample_size
        self.cycles = args.cycles
        self.init_time = 0
        print('initializing population on evolution_trainer init, maybe not the best strategy')
        self.__initialize_population()

    def derive_from_population(self):
        population = self._construct_action(self.population)
        best_score_index, _ = \
            self._get_best_individual_accuracy(self.accuracies)
        best_structure = self.form_gnn_info(population[best_score_index])
        print("[DERIVE] Best Structure:", str(best_structure))
        # train from scratch to get the final score
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        test_scores_list = []
        for i in range(10):  # run 10 times to get Mean and Stddev
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print("[DERIVE] Best Results: ", best_structure, ": ",
              np.mean(test_scores_list),
              "+/-", np.std(test_scores_list))

    def _mutate_individual(self, indiv):
        # Choose a random position on the individual to mutate
        position_to_mutate = np.random.randint(len(indiv))
        # This position will receive a randomly chosen index
        # of the search_spaces's list
        # for the action corresponding to that position in the individual
        sp_list = self.search_space[self.action_list[position_to_mutate]]
        indiv[position_to_mutate] = \
            np.random.randint(0, len(sp_list))
        return indiv

    def _get_best_individual_accuracy(self, accs):
        max_acc_index = 0
        max_acc = -1
        for index, acc in enumerate(accs):
            if acc > max_acc:
                max_acc = acc
                max_acc_index = index
        return max_acc_index, max_acc

    def __initialize_population(self):
        print("\n\n===== Evaluating initial random population =====")
        start_initial_population_time = time.time()
        while len(self.population) < self.population_size:
            # print('adding individual #:', len(population))
            individual = self._generate_random_individual()
            ind_actions = self._construct_action([individual])
            gnn = self.form_gnn_info(ind_actions[0])
            _, ind_acc = \
                self.submodel_manager.train(gnn, format=self.args.format)
            print("individual:", individual, " val_score:", ind_acc)
            self.accuracies.append(ind_acc)
            self.population.append(individual)
        end_initial_pop_time = time.time()
        self.init_time = end_initial_pop_time - start_initial_population_time
        print("Time elapsed initializing population: " +
              str(self.init_time))
        print("===== Evaluating initial random population DONE ====")

    def train(self):
        print("\n\n===== Evolution ====")
        start_evolution_time = time.time()
        while self.cycles > 0:
            sample = []  # list with indexes to population individuals
            sample_accs = []  # accuracies of the sampled individuals
            while len(sample) < self.sample_size:
                candidate = np.random.randint(0, len(self.population))
                sample.append(self.population[candidate])
                sample_accs.append(self.accuracies[candidate])

            # Get best individual on sample to serve as parent
            max_sample_acc_index, max_sample_acc = \
                self._get_best_individual_accuracy(sample_accs)
            parent = sample[max_sample_acc_index]
            # print('parent: ', parent)
            child = parent.copy()
            child = self._mutate_individual(child)
            # print('child: ', child)
            child_actions = self._construct_action([child])
            gnn = self.form_gnn_info(child_actions[0])
            _, child_acc = \
                self.submodel_manager.train(gnn, format=self.args.format)
            # print('child acc: ', child_acc)
            print("parent: ", str(parent), " val_score: ", str(max_sample_acc),
                  "| child: ", str(child), ", val_score: ", str(child_acc))
            self.accuracies.append(child_acc)
            self.population.append(child)
            if self.cycles % self.args.eval_cycle == 0:
                self.derive_from_population()
            # Remove oldest individual (Aging/Regularized evolution)
            self.population.popleft()
            self.accuracies.popleft()
            print("[POPULATION STATS] Mean/Median/Best: ",
                  np.mean(self.accuracies),
                  np.median(self.accuracies),
                  np.max(self.accuracies))
            self.cycles -= 1
        end_evolution_time = time.time()
        total_evolution_time = end_evolution_time - start_evolution_time
        print('Time spent on evolution: ' +
              str(total_evolution_time))
        print('Total elapsed time: ' +
              str(total_evolution_time + self.init_time))
        print("===== Evolution DONE ====")

    def derive(self, sample_num=None):
        self.derive_from_population()
