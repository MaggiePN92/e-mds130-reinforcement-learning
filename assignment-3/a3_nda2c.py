import gym
import torch
import numpy as np
import multiprocessing as mp
from a3_a2c import A2C
import copy


class NDA2C(A2C):       
    def partial_train(self, i, counter, params, info):
        env = copy.deepcopy(self.env)
        state = env.reset()
        score = 0
        ecounter = 0
        gamma = params["gamma"]

        for e in range(params['epochs']):
            done = False
            values, probs, rewards = [], [], []
            G = torch.Tensor([0])
            
            for step in range(params["nstep"]):
                if done: break
                
                value = self.critic(torch.from_numpy(state).float())
                policy = self.actor(torch.from_numpy(state).float())                
                action = self.choose_action(policy)

                state, reward, done, _ = env.step(action)
                values.append(value)                
                probs.append(policy[action])
                rewards.append(reward)

                if done:
                    if ecounter % np.round(params['epochs']/10) == 0:
                        print('worker: {}, epoch:{}, episode: {:d}, score: {:.2f}'.format(
                            i, e, ecounter, score))
                    # Store the progress for each episode
                    info[ecounter] = {'score': info[ecounter]['score'] + score, 'count': info[ecounter]['count'] + 1} if info.get(
                        ecounter) else {'score': score, 'count': 1}
                    state = env.reset()
                    score = 0
                    ecounter += 1
                else:
                    score += reward
                    G = value.detach()

            values = torch.concat(values).flip(
                dims=(0, )).view(-1)                      
            probs = torch.stack(probs).flip(
                dims=(0, )).view(-1)
            rewards = torch.Tensor(rewards).flip(
                dims=(0, )).view(-1)            

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

        counter.value = counter.value + 1            

    def train(self, params):
        self.actor.share_memory()
        self.critic.share_memory()
        
        processes = []

        counter = mp.Value('i', 0)
        info = mp.Manager().dict()
        
        for i in range(params['workers']):
            p = mp.Process(target=self.partial_train, args=(
                i, counter, params, info))
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
        'epochs': 1500,  
        'gamma': 0.99,
        'nstep': 50,
        'workers': mp.cpu_count()
    }

    info = agent.train(training_params)
    #agent.partial_train(0, 0, training_params, info={})
    info_ = np.array([(k, v['score'] / v['count']) for k, v in info.items()])
    
    agent.plot(info_)
    agent.test()
