import gym
import torch
import numpy as np

import sys
from l5_a2c import A2C

# Online training will produce unreliable results

# Inherits class A2C
class A2COnline(A2C):
    # Trains the agent
    def train(self, epochs=1000, gamma=0.99):
        # Stores the info from the training process
        info = {}
        # Stores the total reward or scores per epoch
        scores = []
        for e in range(epochs):
            # At start of each epoch, reset the state of the environment
            state_ = self.env.reset()
            # A flag to check if the episode has ended
            done = False
            # Store the score per episode
            score = 0            
            # Maximum number of moves allowed during the training of an episode
            maxMoves = 200
            # Continue an episode if it hasnt ended of the maximim number of moves hasnt expired
            while not done and maxMoves > 0:
                # Decrement the maximum number of allowed moves in an episode
                maxMoves -= 1
                # Calculate the probs of each action for a state
                policy = self.actor(torch.from_numpy(
                    state_).float())
                # Choose an action based on their probs.
                action = np.random.choice(
                    self.numActionSpace, p=policy.detach().numpy())
                # Execute an action in the environment
                nextState_, reward, done, _ = self.env.step(action)

                # Store the received reward
                score += reward

                # Calculate the value for a state
                value = self.critic(torch.from_numpy(state_).float())
                # Calculate the value for the next state. Since we are doing online training we do not have the return for the next state, 
                # therefore we bootstap / predict the value for the next step
                nextValue = torch.Tensor([0.0]) if done else self.critic(
                    torch.from_numpy(nextState_).float())

                # Value Loss
                criticLoss = self.criticLossFun(
                    value, reward + gamma * nextValue.detach()) 

                # Policy Loss
                advantage = reward + gamma * nextValue.item() - value.item()
                actorLoss = self.actorLossFun(policy[action], advantage) 

                #Backpropagate value
                self.criticOptim.zero_grad()
                criticLoss.backward()
                self.criticOptim.step()

                #Backpropagate policy
                self.actorOptim.zero_grad()
                actorLoss.backward()
                self.actorOptim.step()

                state_ = nextState_                
            # Store the total score for the episode
            scores.append(score)

            # Print the progress
            if e % np.round(epochs/10) == 0:
                print('episode: {:d}, score: {:.2f}'.format(e, scores[e]))
            info[e] = scores[e]

        return info


if __name__ == '__main__':
    # Create the environment
    env = gym.make('CartPole-v1')    
    # Instantiate the agent class.
    agent = A2COnline(env)
    # Train the agent
    info = agent.train(1000)
    # Convert to an numpy array with epochs at axis=0 and scores at axis=1
    info_ = np.array(list(info.items()))
    # Plot the scores
    agent.plot(info_)
    # Run a test
    agent.test()
