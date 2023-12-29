# imports
import numpy as np
# for visualization
import matplotlib.pyplot as plt
import math

import torch


class Hopfield_Net:  # network class
    # init ialize network variables and memory
    def __init__(self, input, p_w=1.):

        # patterns for network training / retrieval
        inputs = np.array(input)
        if inputs.min() == 0:
            inputs[inputs < 0.5] = -1
            inputs[inputs >= 0.5] = 1

        self.memory = inputs
        # single vs. multiple memories
        if self.memory.size > 1:
            self.n = self.memory.shape[1]
        else:
            self.n = len(self.memory)
        # network construction
        self.state = np.random.randint(-2, 2, (self.n, 1))  # state vector
        self.weights = np.zeros((self.n, self.n))  # weights vector
        self.energies = []  # container for tracking of energy
        self.weight_mask = np.random.rand(self.n,self.n)<p_w
        self.p_w = p_w

    def network_learning(self):  # learn the pattern / patterns
        self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory  # hebbian learning
        np.fill_diagonal(self.weights, 0)
        if self.p_w < 1:
            self.weights = self.weights * self.weight_mask


    def update_network_state(self, n_update):  # update network
        for neuron in range(n_update):  # update n neurons randomly
            self.rand_index = np.random.randint(0, self.n)  # pick a random neuron in the state vector
            # Compute activation for randomly indexed neuron
            self.index_activation = np.dot(self.weights[self.rand_index, :],
                                           self.state)
            # threshold function for binary state change
            if self.index_activation < 0:
                self.state[self.rand_index] = -1
            else:
                self.state[self.rand_index] = 1

    def retrival_pattern(self, query):
        if query.min() == 0:
            query[query < 0.5] = -1
            query[query >= 0.5] = 1
        restored = np.dot(self.weights, query)
        restored[restored < 0] = 0
        restored[restored > 0] = 1
        return restored

    def compute_energy(self):  # compute energy
        self.energy = -0.5 * np.dot(np.dot(self.state.T, self.weights), self.state)
        self.energies.append(self.energy)



import numpy as np
import random


class Hopfield:
    def __init__(self, raw_data, m, p, precision,p_w=1):
        ''' Neurons takes binary value 0 or 1.
        args:
           N (int): dimension of the net
           memories (nparray): memory patterns with dimension [number of examples, N]
        returns: None
        '''
        self.W = 1/m*(raw_data.to(precision)-p) @ (raw_data.to(precision).T -p)
        self.W = self.W - torch.eye(m).to('cuda') * self.W.diag()
        self.threshold = p
        self.p_w = p_w
        if p_w < 1:
            self.W =  self.W * (torch.rand(m,m).cuda() < p_w)


    def update(self, num_updates, query_data,precision, threshold=None):
        ''' Update network
        Args:
           num_updates (int): update of updates to perform
           rule (str): update rule.
              'energy_diff': energy difference
              'field': flip by effective field
        Returns:
        '''
        if threshold is None:
            threshold = self.threshold
        new_s = (torch.sign(query_data.to(precision).T @ self.W.to(precision) - threshold) + 1)/2
        for iter in range(num_updates):
            old_s = new_s
            new_s = (torch.sign(old_s.to(precision)  @ self.W.to(precision) - threshold) + 1)/2
            err_sums = (old_s - new_s).abs().sum()
            if err_sums < 1 and iter > 5:
                break
                torch.cuda.empty_cache()
        return new_s.T


def addNoise(X, prob=0.05):
    Xnew = X.copy()
    for i in range(len(Xnew)):
        if random.random() < prob:
            Xnew[i] *= -1
    return Xnew
