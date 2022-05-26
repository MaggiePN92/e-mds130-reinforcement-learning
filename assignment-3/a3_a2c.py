import torch
import numpy as np
from matplotlib import pyplot as plt


class A2C:
    def __init__(self, env, h1, lr):
        torch.seed = 999
        torch.manual_seed(torch.seed)
        np.random.seed(torch.seed)

        self.env = env
        self.env.seed = torch.seed

        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.actor = self.init_actor(h1)
        self.critic = self.init_critic(h1)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss_func = torch.nn.MSELoss()


    def init_actor(self, h1):
        actor = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, self.n_actions),
            torch.nn.Softmax(dim=-1)
        )
        return actor

    def init_critic(self, h1):
        critic = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, 1),
        )
        return critic

    def actor_loss(self, probs, adv):
        loss =  -1 * torch.sum(torch.log(probs) * adv)
        return loss

    def choose_action(self, policy):
        action = np.random.choice(
            self.n_actions,
            p=policy.detach().numpy()
        )
        return action

    def calc_returns(self, last_return, rewards, gamma):
        returns = []
        for ret in range(rewards.shape[0]):
            return_ = rewards[ret] + gamma * last_return
            returns.append(return_)
        return torch.stack(returns).view(-1)

    def print_progress(self, ep_counter, epoch, max_epochs, worker_id, score):
        if ep_counter % np.round(max_epochs/10) == 0:
            print(f'worker: {worker_id}, epoch:{epoch}, episode: {ep_counter:d}, score: {score:.2f}')

    def store_progress(self, info, ep_counter, score):
        info[ep_counter] = {
            'score': info[ep_counter]['score'] + score, 'count': info[ep_counter]['count'] + 1} if info.get(
                        ep_counter) else {'score': score, 'count': 1}
        return info

    def train(self):
        raise NotImplementedError

    def test(self, render=False, max_moves=500, ):
        state = self.env.reset()
        done = False
        score = 0

        for _ in range(max_moves):
            if done: break
            if render:
                self.env.render()

            action = self.choose_action(state)

            state, reward, done, _ = self.env.step(action)
            score += reward
        
        print(f'reward: {score}')


    def plot(self, info):
        info.sort(axis=0)
        x, y = info[:, 0], info[:, 1]

        plt.plot(x, y)
        plt.title('Scores over episodes:')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.show()
