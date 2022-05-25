import gym
import torch
import numpy as np
import multiprocessing as mp
from a3_a2c import A2C
import copy


class DA2C(A2C):
    
    def partialTrain(self, i, counter, params, info):
        env = copy.deepcopy(self.env)
        scores = []

        for e in range(params['epochs']):
            state_ = env.reset()
            values, probs, rewards = [], [], []
            done = False
            score = 0
            maxMoves = 200            
            
            print("\n")
            print(f"Epoch {e}")

            for _ in range(maxMoves):
                if done: break
            
                value = self.critic(torch.from_numpy(state_).float())               
                policy = self.actor(torch.from_numpy(
                    state_).float())                
                action = np.random.choice(
                    self.numActionSpace, p=policy.detach().numpy())
                
                print(f"For episode {_} ")
                print(f"Action: {action}")

                state_, reward, done, _ = env.step(action)
                
                values.append(value)                
                probs.append(policy[action])
                rewards.append(0 if done else reward)
                score += reward

            
            scores.append(score)            
            # Convert to tensors and also reverse the values, probs and rewards. 
            # We reverse them as it simplifies the calculations for the returns
            values = torch.concat(values).flip(
                dims=(0, )).view(-1)                      
            probs = torch.stack(probs).flip(
                dims=(0, )).view(-1)
            rewards = torch.Tensor(rewards).flip(dims=(0, )).view(-1)            

            # Store the returns
            Returns = []
            # Intermediate variable to store the return per step
            ret_ = torch.Tensor([0])

            # iterate through the steps
            for r in range(rewards.shape[0]):
                # The first intermediate return is for the last step as we had reversed them. The next
                # is the one prior to that and so forth. We are also multiplying the gamma to the future
                # return. At the first iteration, the return value is zero as its the last step. Therefore
                # its return will be the reward itself. For all iterations other than the first ret_ stores
                # the Return for the future and gamma applies the decay factor to future rewards
                ret_ = rewards[r] + params['gamma'] * ret_
                Returns.append(ret_)
            
            # Convert to a tensor
            Returns = torch.stack(Returns).view(-1)
                    
            criticLoss = self.criticLossFun(
                values, Returns)

            # Calculate the advantage based on what was returned by the actor and what the critic said
            # would be the value for the state.
            advantage = Returns - values.detach()
            actorLoss = self.actor_loss(probs, advantage)            
            
            
            self.criticOptim.zero_grad()
            criticLoss.backward()
            self.criticOptim.step()

            self.actorOptim.zero_grad()
            actorLoss.backward()
            self.actorOptim.step()            
                            

            # Print the progress
            if e % np.round(params['epochs']/10) == 0:
                print('worker: {}, episode: {:d}, score: {:.2f}'.format(
                    i, e, scores[e]))
            
            # Store the progress for each episode
            info[e] = {'score': info[e]['score'] + scores[e], 'count': info[e]['count'] + 1} if info.get(
                e) else {'score': scores[e], 'count': 1}

        # Counter is globally shared between all running processes
        counter.value = counter.value + 1            




if __name__ == '__main__':
    # Create the environment
    env = gym.make('MountainCar-v0')
    # Instantiate the agent class.
    agent = DA2C(env, 350)
    
    ##### DEBUG TRAINER ######
    params = {
        'epochs': 1000,  
        'gamma': 0.99
    }


    info = agent.partialTrain(0, 0, params, info={})
    # Averages the scores per epoch. Also ensure that the epochs are at axis=0 and scores at axis=1
    info_ = np.array([(k, v['score'] / v['count']) for k, v in info.items()])
    # Plot the scores
    agent.plot(info_)    
    # Run a test
    agent.test()
