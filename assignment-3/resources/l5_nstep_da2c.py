import gym
import torch
import numpy as np
import multiprocessing as mp

import sys
from l5_a2c import A2C
import copy


# Inherits class A2C
class NStepDA2C(A2C):       
    def partialTrain(self, i, counter, params, info):
        # Make a copy of the environment for each worker
        env = copy.deepcopy(self.env)
        
        # Specify the loss functions for each worker
        criticLossFun = torch.nn.MSELoss()
        # Since we are trying batch processing, we need to sum up the losses
        actorLossFun = lambda probs, advantage: -1 * torch.sum(torch.log(probs) * advantage)

        # <--- Since we might exit an episode before it finishes due to nstep, we initialize the following variable outside the epochs loop.
        # <--- Reset the environment at the start
        state_ = env.reset()
        # <--- Flag to check if an episode is done. 
        done = False
        # <--- Stores the total score for each episode. 
        score = 0
        # <--- Max number of allowed moves per episode. 
        maxMoves = 200        
        # <--- Counter for the number of episodes
        ecounter = 0
        for e in range(params['epochs']):
            done = False
            # Will store the values, probs of actions and respective rewards for each episode
            values, probs, rewards = [], [], []                                                
            # Will keep track of number of steps in an episode
            j = 0
            # Will be storing the return as well. It is initialized to 0
            G = torch.Tensor([0])
            # <--- Continue the episode until done flag is True or maximum number of allowed moves have expired
            
            while not done and (j < params['nstep'] or maxMoves > 0):
                # Increment the number of steps
                j += 1
                # Calculate the value of the state
                value = self.critic(torch.from_numpy(state_).float())               
                # Calculate the probs of each action for the state
                policy = self.actor(torch.from_numpy(
                    state_).float())                
                # Choose an action based on their prob. dist.
                action = np.random.choice(
                    self.numActionSpace, p=policy.detach().numpy())
                # Execute an action in the environment
                nextState_, reward, done, _ = env.step(action)
                # Store the value of the state, the prob. of the choosen action and the received reward
                values.append(value)                
                probs.append(policy[action])

                if done:
                    rewards.append(-10.0)
                    # <--- Since we are doing nstep learning, we exit in between an episode even when it hasnt finished                    
                    # Print the progress
                    if ecounter % np.round(params['epochs']/10) == 0:
                        print('worker: {}, epoch:{}, episode: {:d}, score: {:.2f}'.format(
                            i, e, ecounter, score))
                    # Store the progress for each episode
                    info[ecounter] = {'score': info[ecounter]['score'] + score, 'count': info[ecounter]['count'] + 1} if info.get(
                        ecounter) else {'score': score, 'count': 1}

                    # We reset the environment only when the game is actually over
                    state_ = env.reset()                    
                    score = 0
                    maxMoves = 200
                    ecounter += 1
                else:
                    # <--- Set the next state as current state
                    state_ = nextState_
                    rewards.append(reward)
                    # <--- If we are continuing the same episode    
                    # Update the total reward received for this episode
                    score += reward
                    # Decrement the maximum number of allowed moves in an episode
                    maxMoves -= 1
                    # <--- Everytime, the episode isn't over we replace the current return stored in G as the current value for the state
                    G = value.detach()
                                
                        
            # Convert to tensors and also reverse the values, probs and rewards. We reverse them as it simplifies the calculations for the returns
            values = torch.concat(values).flip(
                dims=(0, )).view(-1)                      
            probs = torch.stack(probs).flip(
                dims=(0, )).view(-1)
            rewards = torch.Tensor(rewards).flip(dims=(0, )).view(-1)            

            # Store the returns
            Returns = []
            # <--- Intermediate variable to store the return per step. We also set the intermediate return as the value of G. 
            # If the episode has ended it will be zero, else when the episode was stopped due to reaching nsteps we store the current value of the state where it was stopped
            ret_ = G

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

        # Counter is globally shared between all running processes
        counter.value = counter.value + 1            

    # Trains the agent
    def train(self, epochs=1000, gamma=0.99, nstep=10):
        # With share memory, the parameters will be shared among all workers so that they can do live updates
        self.actor.share_memory()
        self.critic.share_memory()
        # A list to store the initiated processes
        processes = []
        params = {
            'epochs': epochs,  
            'gamma': gamma,
            # <--- Specify the number of steps an episode should run
            'nstep': nstep,
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
    agent = NStepDA2C(env)
    # Train the agent
    info = agent.train(1000)
    # Averages the scores per epoch. Also ensure that the epochs are at axis=0 and scores at axis=1
    info_ = np.array([(k, v['score'] / v['count']) for k, v in info.items()])
    # Plot the scores
    agent.plot(info_)    
    # Run a test
    agent.test()