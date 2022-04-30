from agent import AgentQLearning
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gym


class QDeepLearningExpReplay(AgentQLearning):
    """
    Q Deep Learning with experience replay. Experience replay is, instead of online training, to perform backprop
    batch wise. This helps with catastrophic forgetting.
    """
    def __init__(self, lr, gamma, max_eps, min_eps, env=gym.make("CartPole-v1"), n_actions=2, state_dim=4,
                 h1_in=120, h1_out=100, optimizer=torch.optim.Adam, loss_func=torch.nn.MSELoss):
        super().__init__(lr, gamma, max_eps, min_eps, env, n_actions, state_dim, h1_in, h1_out, optimizer,
                         loss_func)

    def train(self, epochs=1000, mem_size=1000, batch_size=200, max_num_games=150):
        assert mem_size > batch_size, \
            f"Batch size = {batch_size} should be smaller than the size of the replay memory, mem_size = {mem_size}."

        losses = []
        epsilon = self.max_eps
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
                        next_q_vals = self.model(next_state_batch)


                    y_hat = reward_batch +  self.gamma * ((1 - done_batch) * torch.max(next_q_vals, dim=1)[0])


                    y = q_vals.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                    loss = self.loss_func(y, y_hat)
                    losses.append(loss.item() / batch_size)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if done:
                    break


            if epsilon > self.min_eps:
                epsilon -= (1 / 1000)

            # Print the progress
            if ((e % np.round(epochs / 10) == 0)):
                error = np.mean(losses)
                print(f'epoch: {e:d}, error: {error:.2f}')
            plt.plot(losses)


    def train_torch(self, epochs=1000, mem_size=1000, batch_size=200, max_num_games=150):
        losses = []
        epsilon = self.max_eps

        state_batch = torch.empty((batch_size, self.state_dim))
        next_state_batch = torch.empty((batch_size, self.state_dim))

        action_batch = torch.empty((batch_size, 1))
        reward_batch = torch.empty((batch_size, 1))
        done_batch = torch.empty((batch_size, 1))

        c = 0

        for _ in tqdm(range(epochs)):
            state = torch.from_numpy(self.env.reset()).to(self.device).float()
            done = False

            for g in range(max_num_games):
                q_vals = self.model(state)

                if np.random.random() < epsilon:
                    action = np.random.choice(self.n_actions)
                else:
                    action = torch.argmax(q_vals).item()

                next_state, reward, done, _ = self.env.step(action)

                if c < batch_size:
                    state_batch[c] = state
                    next_state_batch[c] = torch.from_numpy(next_state)

                    action_batch[c] = action
                    reward_batch[c] = reward
                    done_batch[c] = done
                    c += 1

                    if done:
                        break

                    continue


                elif c >= mem_size:
                    idx = c - (mem_size * (c // mem_size))

                    state_batch[idx] = state
                    next_state_batch[idx] = torch.from_numpy(next_state)

                    action_batch[idx] = action
                    reward_batch[idx] = reward
                    done_batch[idx] = done

                else:
                    state_batch = torch.cat((state_batch, torch.unsqueeze(state, dim=0)), dim=0)

                    next_state = torch.from_numpy(next_state)
                    next_state_batch = torch.cat((next_state_batch, torch.unsqueeze(next_state, dim=0)), dim=0)

                    action_batch = torch.cat((action_batch, torch.unsqueeze(torch.tensor([action]), dim=0)), dim=0)
                    reward_batch = torch.cat((reward_batch, torch.unsqueeze(torch.Tensor([reward]), dim=0)), dim=0)
                    done_batch = torch.cat((done_batch, torch.unsqueeze(torch.Tensor([done]), dim=0)), dim=0)

                c += 1
                random_indices = torch.randperm(action_batch.size(0))[:batch_size]

                q_vals = self.model(state_batch[random_indices])

                with torch.no_grad():
                    next_q_vals = self.model(next_state_batch[random_indices])

                y_hat = reward_batch[random_indices].squeeze() + 0.99 * (
                            (1 - done_batch[random_indices]).squeeze() * torch.max(next_q_vals, dim=1)[0])
                y = q_vals.gather(dim=1, index=action_batch[random_indices].long()).squeeze()

                loss = self.loss_func(y, y_hat)
                losses.append(loss.item() / batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if done:
                    break

        if epsilon > self.min_eps:
            epsilon -= (1 / 1000)

            # Print the progress
        if ((c % np.round(epochs / 10) == 0)):
            error = np.mean(losses)
            print(f'epoch: {c:d}, error: {error:.2f}')
        plt.plot(losses)


def main():
    # a = QDeepLearningExpReplay(1e-3, 0.99, 1, 0.001)
    # a.train(epochs=300)
    # print(np.mean(a.test(num_games=100)))

    b = QDeepLearningExpReplay(1e-3, 0.99, 1, 0.001)
    b.train_torch(epochs=300)
    print("----")
    print(np.mean(b.test(num_games=100)))


if __name__ == "__main__":
    main()
