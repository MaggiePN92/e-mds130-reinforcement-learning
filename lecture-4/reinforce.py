import torch
from torch import nn
import numpy as np


class Reinforce:
    def __init__(self, env, n_actions, state_dim, h1_size, lr=3e-3, optimizer=torch.optim.Adam):
        self.env = env
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.h1_size = h1_size
        self.model = self.init_network()
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), self.lr)

    def init_network(self):
        network = torch.nn.Sequential(
            nn.Linear(self.state_dim, self.h1_size),
            nn.ReLU(),
            nn.Linear(self.h1_size, self.n_actions),
            nn.Softmax(dim=0)
        )
        return network

    def action_pred(self, state):
        act_prob = self.model(state).float()
        action = np.random.choice(self.n_actions, act_prob)
        return action

    def calc_expected_return(self, transition_len, reward_batch, gamma):
        expected_return_batch = []
        for i in range(transition_len):
            g_val = 0
            power = 0
            for j in range(i, transition_len):
                g_val += (gamma**power)*reward_batch[j]
                power += 1
        expected_return_batch = [ ]
        return expected_return_batch

    def train(self, gamma, horizon, max_trajectories):
        state = self.env.reset()
        done = False
        transitions = []
        score = []

        for t in range(max_trajectories):
            action = self.action_pred(torch.from_numpy(state).float())
            prev_state = state
            state, _, done, info = self.env.step(action)
            transitions.append((prev_state, action, t+1))
            if done:
                break

        score.append(len(transitions))
        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))

        # estimate return trajectory
        batch_Gvals = []


    def test(self):
        pass

