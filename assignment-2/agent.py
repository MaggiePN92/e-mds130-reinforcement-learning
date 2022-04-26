import torch
from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt


class AgentQLearning:
    def __init__(self, lr, gamma, max_eps, min_eps, epochs=1000, env=gym.make("CartPole-v1"), n_actions=2, state_dim=4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.model = self.init_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()
        self.lr = lr
        self.gamma = gamma # discount rate
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.epochs = epochs
        self.env = env

    def init_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.n_actions)
        )
        return model

    def train(self):
        losses = []
        epsilon = self.max_eps
        self.model.train()

        for e in tqdm(range(self.epochs)):
            # 1. Init done and state variables
            # 2. Start game
            #    a) Get q_vals
            #    b) Choose and do action, receive reward and new state
            #    c) Calculate y_hat and y, and do backward

            state = torch.from_numpy(self.env.reset()).to(self.device).float()
            done = False

            # starting play
            while not done:
                q_vals = self.model(state)
                q_vals_np = q_vals.data.cpu().numpy()

                if np.random.random() < epsilon:
                    action = np.random.choice(self.n_actions)
                else:
                    action = np.argmax(q_vals_np)

                next_state, reward, done, info = self.env.step(action)
                state = next_state = torch.from_numpy(next_state).to(self.device).float()

                with torch.no_grad():
                    next_q_vals = self.model(next_state)

                y_hat = reward

                if not done:
                    y_hat += (self.gamma * torch.max(next_q_vals))

                y_hat = torch.Tensor([y_hat]).detach().to(self.device)
                y = q_vals.squeeze()[action]

                loss = self.loss_func(y, y_hat)
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epsilon > self.min_eps:
                epsilon -= (1 / 1000)

            # Print the progress
            if e % np.round(self.epochs / 10) == 0:
                error = np.mean(losses)
                print(f'epoch: {e:d}, error: {error:.2f}')

            plt.plot(losses)

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
