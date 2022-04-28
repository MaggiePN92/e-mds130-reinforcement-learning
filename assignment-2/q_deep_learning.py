from agent import AgentQLearning
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym


class QDeepLearning(AgentQLearning):
    def __init__(self, lr, gamma, max_eps, min_eps, env=gym.make("CartPole-v1"), n_actions=2, state_dim=4,
                 h1_in=120, h1_out=100, optimizer=torch.optim.Adam, loss_func=torch.nn.MSELoss):
        super().__init__(lr, gamma, max_eps, min_eps, env, n_actions, state_dim, h1_in, h1_out, optimizer,
                         loss_func)

    def train(self, epochs):
        losses = []
        epsilon = self.max_eps
        self.model.train()

        for e in tqdm(range(epochs)):
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
            if e % np.round(epochs / 10) == 0:
                error = np.mean(losses)
                print(f'epoch: {e:d}, error: {error:.2f}')

            plt.plot(losses)


def main():
    a = QDeepLearning(1e-3, 0.99, 1, 0.01)
    a.train(epochs=10)

if __name__ == "__main__":
    main()