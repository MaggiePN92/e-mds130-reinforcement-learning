import torch
from torch import nn
import numpy as np
import gym


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
        print(act_prob)
        action = np.random.choice(self.n_actions, p=act_prob.data.numpy())
        return action

    def calc_expected_return(self, reward_batch, gamma):
        transition_len = max(reward_batch.shape)
        expected_return_batch = torch.empty((transition_len, 1))
        for i in range(transition_len):
            g_val = 0
            power = 0
            for j in range(i, transition_len):
                g_val += (gamma**power)*reward_batch[j]
                power += 1
            expected_return_batch[i] = g_val
        expected_return_batch /= expected_return_batch.max()
        return expected_return_batch

    def train(self, gamma=0.99, horizon=500, max_trajectories=500):
        score = []
        losses = []

        for trajectory in range(max_trajectories):
            state = self.env.reset()

            state_batch = torch.empty(size=(max_trajectories, 4))
            action_batch = torch.empty(size=(max_trajectories, 1))
            reward_batch = torch.empty(size=(max_trajectories, 1))

            for t in range(horizon):
                action = self.action_pred(torch.from_numpy(state).float())
                prev_state = state
                state, _, done, info = self.env.step(action)

                # prev_state -> state_batch
                state_batch[t] = torch.from_numpy(prev_state)
                # action -> action_batch
                action_batch[t] = action
                # reward (=t+1) -> reward_batch
                reward_batch[t] = t+1

                if done:
                    state_batch = state_batch[:max_trajectories]
                    action_batch = action_batch[:max_trajectories]
                    reward_batch = reward_batch[:max_trajectories]
                    break

            score.append(reward_batch.shape[0])
            expected_return_batch = self.calc_expected_return(reward_batch, gamma)

            pred_batch = self.model(state_batch)
            # print(pred_batch.shape)
            # print(pred_batch)
            print(action_batch.shape)
            # print(action_batch)
            prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

            loss = -torch.sum(torch.log(prob_batch) * expected_return_batch)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Trajectory {trajectory} \t loss: {np.mean(losses)} \t score: {np.mean(score[-50:-1])}")

    def test(self):
        pass


def main():
    env = gym.make("CartPole-v1")
    n_actions = 2
    state_dim = 4
    h1_size = 100

    r = Reinforce(env, n_actions, state_dim, h1_size)
    r.train()


if __name__ == "__main__":
    main()
