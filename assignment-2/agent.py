import torch
from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt


class AgentQLearning:
    def __init__(self, lr, gamma, max_eps, min_eps, env=gym.make("CartPole-v1"), n_actions=2, state_dim=4,
                 h1_in=120, h1_out=100, optimizer=torch.optim.Adam, loss_func=torch.nn.MSELoss):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.h1_in = h1_in
        self.h1_out = h1_out
        self.model = self.init_model().to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_func = loss_func()
        self.lr = lr
        self.gamma = gamma # discount rate
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.env = env

    def init_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.h1_in),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1_in, self.h1_out),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1_out, self.n_actions)
        )
        return model

    def test(self, num_games=100):
        avg_rewards = []
        self.model.eval()

        for _ in tqdm(range(num_games)):
            state = torch.from_numpy(self.env.reset()).to(self.device).float()
            done = False
            avg_reward = 0

            while not done:
                q_vals = self.model(state).data.cpu().numpy()

                action = np.argmax(q_vals)
                state, reward, done, info = self.env.step(action)
                state = torch.from_numpy(state).to(self.device).float()

                avg_reward += reward

            avg_rewards.append(avg_reward)

        plt.plot(avg_rewards)

        return avg_rewards
