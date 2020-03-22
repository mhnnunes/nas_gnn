import time
import torch
import numpy as np
from collections import deque
from graphnas.trainer import Trainer


class RandomSearch_Trainer(Trainer):
    """
    This class implements a Random Search method, on the Search Space
    provided to it.
    """
    def __init__(self, args):
        super(RandomSearch_Trainer, self).__init__(args)
        self.args = args
        self.random_seed = args.random_seed
        self.cycles = args.cycles

    def train(self):
        print("\n\n===== Random Search ====")
        start_time = time.time()
        self.best_ind_acc = 0.0
        self.best_ind = []
        while self.cycles > 0:
            individual = self._generate_random_individual()
            ind_actions = self._construct_action([individual])
            gnn = self.form_gnn_info(ind_actions[0])
            _, ind_acc = \
                self.submodel_manager.train(gnn, format=self.args.format)
            print("individual:", individual, " val_score:", ind_acc)
            if ind_acc > self.best_ind_acc:
                self.best_ind = individual.copy()
                self.best_ind_acc = ind_acc
        end_time = time.time()
        total_time = end_time - start_time
        print('Total elapsed time: ' + str(total_time))
        print('[BEST STRUCTURE]', self.best_ind)
        print('[BEST STRUCTURE] Actions: ',
              self._construct_action([self.best_ind]))
        print('[BEST STRUCTURE] Accuracy: ', self.best_ind_acc)
        print("===== Random Search DONE ====")
