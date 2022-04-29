from agent import AgentQLearning
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gym
import copy


class QDLTransfer(AgentQLearning):
    """
    Q Deep Learning with experience replay. Experience replay is, instead of online training, to perform backprop
    batch wise. This helps with catastrophic forgetting.
    """
    def __init__(self, lr, gamma, max_eps, min_eps, env=gym.make("CartPole-v1"), n_actions=2, state_dim=4,
                 h1_in=120, h1_out=100, optimizer=torch.optim.Adam, loss_func=torch.nn.MSELoss):
        super().__init__(lr, gamma, max_eps, min_eps, env, n_actions, state_dim, h1_in, h1_out, optimizer,
                         loss_func)
        self.transfer_network = self.init_transfer_network()

    def init_transfer_network(self):
        transfer_network = copy.deepcopy(self.model)
        transfer_network.load_state_dict(self.model.state_dict())
        return transfer_network

    def train(self, epochs=1000, mem_size=1000, batch_size=200, max_num_games=150, sync_freq=500):
        assert mem_size > batch_size, \
            f"Batch size = {batch_size} should be smaller than the size of the replay memory, mem_size = {mem_size}."

        losses = []
        epsilon = self.max_eps
        transfer_network_sync_freq = 0
        self.model.train()

        # specific for Q learning with exp replay
        Experience = namedtuple("buffer", field_names=["state", "action", "reward", "done", "next_state"])
        replay = deque(maxlen=mem_size)

        for e in tqdm(range(epochs)):
            # 1. Init done and state variables
            # 2. Start game
            #    a) Get q_vals
            #    b) Choose and do action, receive reward and new state
            #    c) If memory is larger than batch size -> select batch_size rows from memory and do next step batched
            #    d) Calculate y_hat and y, and do backward
            state = torch.from_numpy(self.env.reset()).to(self.device).float()
            done = False

            for _ in range(max_num_games):
                transfer_network_sync_freq += 1
                q_vals = self.model(state)

                if np.random.random() < epsilon:
                    action = np.random.choice(self.n_actions)
                else:
                    action = torch.argmax(q_vals).item()

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).to(self.device).float()

                replay.append(Experience(state, action, reward, done, next_state))
                state = next_state

                if len(replay) > batch_size:
                    random_indices = np.random.choice(len(replay), batch_size, replace=False)

                    state_batch = torch.stack([replay[idx].state for idx in random_indices]).to(self.device)
                    next_state_batch = torch.stack([replay[idx].next_state for idx in random_indices]).to(self.device)

                    action_batch = torch.Tensor([replay[idx].action for idx in random_indices]).to(self.device)
                    reward_batch = torch.Tensor([replay[idx].reward for idx in random_indices]).to(self.device)
                    done_batch = torch.Tensor([replay[idx].done for idx in random_indices]).to(self.device)

                    q_vals = self.model(state_batch)

                    with torch.no_grad():
                        next_q_vals = self.transfer_network(next_state_batch)


                    y_hat = reward_batch +  self.gamma * ((1 - done_batch) * torch.max(next_q_vals, dim=1)[0])


                    y = q_vals.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                    loss = self.loss_func(y, y_hat)
                    losses.append(loss.item() / batch_size)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if transfer_network_sync_freq % sync_freq == 0:
                        self.transfer_network.load_state_dict(self.model.state_dict())

                if done:
                    break


            if epsilon > self.min_eps:
                epsilon -= (1 / 1000)

            # Print the progress
            if ((e % np.round(epochs / 10) == 0)):
                error = np.mean(losses)
                print(f'epoch: {e:d}, error: {error:.5f}')
            plt.plot(losses)


def main():
    a = QDLTransfer(1e-3, 0.99, 1, 0.001)
    a.train(epochs=600)
    test_results = a.test(num_games=100)
    print(np.mean(test_results))


if __name__ == "__main__":
    main()
