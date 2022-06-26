import gym
import torch
import numpy as np
import multiprocessing as mp

import sys
from l5_a2c import A2C
import copy


# Inherits class A2C
class DA2C(A2C):       
    def partialTrain(self, i, counter, params, info):
        # Make a copy of the environment for each worker
        env = copy.deepcopy(self.env)
        
        # Specify the loss functions for each worker
        criticLossFun = torch.nn.MSELoss()
        # Since we are trying batch processing, we need to sum up the losses
        actorLossFun = lambda probs, advantage: -1 * torch.sum(torch.log(probs) * advantage)

        # Stores the scores per epoch for each worker
        scores = []

        for e in range(params['epochs']):
            # Reset the environment at the start of each epoch
            state_ = env.reset()
            # Will store the values, probs of actions and respective rewards for each episode
            values, probs, rewards = [], [], []
            # Flag to check if an episode is done
            done = False
            # Stores the total score for each episode
            score = 0
            # Max number of allowed moves per episode
            maxMoves = 200            
            # Continue the episode until done flag is True or maximum number of allowed moves have expired
            while not done and maxMoves > 0:
                # Decrement the maximum number of allowed moves in an episode
                maxMoves -= 1
                # Calculate the value of the state
                value = self.critic(torch.from_numpy(state_).float())               
                # Calculate the probs of each action for the state
                policy = self.actor(torch.from_numpy(
                    state_).float())                
                # Choose an action based on their prob. dist.
                action = np.random.choice(
                    self.numActionSpace, p=policy.detach().numpy())
                # Execute an action in the environment
                state_, reward, done, _ = env.step(action)
                # Store the value of the state, the prob. of the choosen action and the received reward
                values.append(value)                
                probs.append(policy[action])
                rewards.append(-10.0 if done else reward)
                # Update the total reward received for this episode
                score += reward

            # After an episode ends, store its total score / reward
            scores.append(score)            
            # Convert to tensors and also reverse the values, probs and rewards. We reverse them as it simplifies the calculations for the returns
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
                # The first intermediate return is for the last step as we had reversed them. The next is the one prior to that and so forth
                # We are also multiplying  the gamma to the future return. At the first iteration, the return value is zero as its the last step. Therefore its return will be the reward itself
                # For all iterations other than the first ret_ stores the Return for the future and gamma applies the decay factor to future rewards
                ret_ = rewards[r] + params['gamma'] * ret_
                Returns.append(ret_)
            
            # Convert to a tensor
            Returns = torch.stack(Returns).view(-1)
                    
            # Value Loss
            criticLoss = criticLossFun(
                values, Returns)

            # Policy Loss
            # Calculate the advantage based on what was returned by the actor and what the critic said would be the value for the state
            advantage = Returns - values.detach()
            actorLoss = actorLossFun(probs, advantage)            
            
            #Backpropagate value
            self.criticOptim.zero_grad()
            criticLoss.backward()
            self.criticOptim.step()

            #Backpropagate policy
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

    # Trains the agent
    def train(self, epochs=1000, gamma=0.99):
        # With share memory, the parameters will be shared among all workers so that they can do live updates
        self.actor.share_memory()
        self.critic.share_memory()
        # A list to store the initiated processes
        processes = []
        params = {
            'epochs': epochs,  
            'gamma': gamma,
            # Number of cores on your machine
            'workers': mp.cpu_count()
        }
        # A global counter to identify the workers running the processes
        counter = mp.Value('i', 0)
        # A global dict to store the info from the training process 
        info = mp.Manager().dict()
        
        # Run as may processes as there are workers
        for i in range(params['workers']):
            # Each process will run the partialTrain function. It receives the worker counter, params and the info dict as arguments
            p = mp.Process(target=self.partialTrain, args=(
                i, counter, params, info))
            # Starts a new process that runs the worker function
            p.start()
            processes.append(p)

        for p in processes:
            # Joins each process to wait for it to finish before returning to the main process
            p.join()
        
        for p in processes:
            # Ensures that each process is terminated
            p.terminate()
        # Shows the total number of workers and the exit code for process 1 which will be zero if everything went ok
        print(counter.value, processes[1].exitcode)
        return info

if __name__ == '__main__':
    # Create the environment
    env = gym.make('CartPole-v1')
    # Instantiate the agent class.
    agent = DA2C(env)
    # Train the agent
    info = agent.train(1000)
    # Averages the scores per epoch. Also ensure that the epochs are at axis=0 and scores at axis=1
    info_ = np.array([(k, v['score'] / v['count']) for k, v in info.items()])
    # Plot the scores
    agent.plot(info_)    
    # Run a test
    agent.test()
