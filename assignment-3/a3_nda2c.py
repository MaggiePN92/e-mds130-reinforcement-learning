import gym
import torch
import numpy as np
import multiprocessing as mp
from a3_a2c import A2C
import copy


class NDA2C(A2C):
    def partial_train(self, i, params, info):
        env = copy.deepcopy(self.env)
        state = env.reset()
        ep_counter = 0
        score = 0
        gamma = params["gamma"]
        epochs = params['epochs']
        steps = params["nstep"]

        for e in range(epochs):
            done = False
            values, probs, rewards = [], [], []
            G = torch.Tensor([0])

            for step in range(steps):
                if done: break

                value = self.critic(torch.from_numpy(state).float())
                policy = self.actor(torch.from_numpy(state).float())
                action = self.choose_action(policy)

                state, reward, done, _ = env.step(action)
                
                values.append(value)
                probs.append(policy[action])
                rewards.append(reward)
                score += reward
                
                if done:
                    self.print_progress(ep_counter, e, epochs, i, score)
                    info = self.store_progress(info, ep_counter, score)
                    state = env.reset()
                    ep_counter += 1
                    score = 0
                else:
                    G = value.detach()

            values = torch.concat(values).flip(
                dims=(0, ))
            probs = torch.stack(probs).flip(
                dims=(0, ))
            rewards = torch.Tensor(rewards).flip(
                dims=(0, ))

            returns = self.calc_returns(G, rewards, gamma)

            critic_loss = self.critic_loss_func(values, returns)

            advantage = returns - values.detach()
            actor_loss = self.actor_loss(probs, advantage)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

    def train(self, params):
        self.actor.share_memory()
        self.critic.share_memory()

        processes = []
        info = mp.Manager().dict()
        workers = params["workers"]

        for i in range(workers):
            p = mp.Process(target=self.partial_train, args=(
                i, params, info))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for p in processes:
            p.terminate()

        return info


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    agent = NDA2C(env, 150, 1e-3)

    training_params = {
        'epochs': 300,  
        'gamma': 0.99,
        'nstep': 50,
        'workers': mp.cpu_count()
    }
    info = agent.train(training_params)

    agent.plot(info)
    agent.test()
