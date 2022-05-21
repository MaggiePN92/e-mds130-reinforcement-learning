import torch
import numpy as np
from matplotlib import pyplot as plt

class A2C:
    def __init__(self, env):
        torch.seed = 42
        torch.manual_seed(torch.seed)
        np.random.seed(torch.seed)
        
        self.env = env
        self.env.seed = torch.seed
        self.numStateSpace = self.env.observation_space.shape[0]
        self.numActionSpace = self.env.action_space.n
        self.actorOptim = torch.optim.Adam(self.actor.parameters(), lr=0.001)        
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.criticLossFun = torch.nn.MSELoss()

    def init_actor(self):
        actor = torch.nn.Sequential(
            torch.nn.Linear(self.numStateSpace, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, self.numActionSpace),
            torch.nn.Softmax(dim=-1)
        )
        return actor

    def init_critic(self):
        critic = torch.nn.Sequential(
            torch.nn.Linear(self.numStateSpace, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
            # torch.nn.Tanh()
        )
        return critic

    
    # We may have defined it as a lambda function inside the constructor as well, 
    # however, doing so will create issues with multiprocessing if this class is inherited by others
    def actorLossFun(self, probs, advantage): 
        return - 1 * torch.log(probs) * advantage

    def train(self):
        raise NotImplementedError

    def test(self, render=False):
        state_ = self.env.reset()
        done = False
        maxMoves = 500
        score = 0
        
        for _ in range(maxMoves):
            if done: break
            if render:
                self.env.render()

            policy = self.actor(torch.from_numpy(
                state_).float())

            action = np.random.choice(
                len(policy), p=policy.detach().numpy())

            state_, reward, done, _ = self.env.step(action)    
            score += reward
        
        print(f'reward: {score}')
    
    
    def plot(self, info_):
    
        info_.sort(axis=0)
        x, y = info_[:, 0], info_[:, 1]
        
        plt.plot(x, y)
        plt.title('Scores')
        plt.xlabel('episode')
        plt.ylabel('score')        
        plt.show()
