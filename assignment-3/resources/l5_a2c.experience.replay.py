import gym
import torch
import numpy as np

import sys
from l5_a2c import A2C
from collections import deque

# Batch Training in this way is not a good idea as temporal dependency is lost

# Inherits class A2C
class A2CBatch(A2C):
    def __init__(self, env):
        super().__init__(env)
        # Overrides the loss function for the actor. Since we are trying batch processing, we need to sum up the losses
        self.actorLossFun = lambda probs, advantage: - \
            1 * torch.sum(torch.log(probs) * advantage)

    # Trains the agent            
    def train(self, epochs=200, gamma=0.99, memory=500, batch=200):
        # Stores the info from the training process
        info = {}
        # Stores the total reward or scores per epoch
        scores = []
        # The replay buffer for Q-learning with experience replay
        replay = deque(maxlen=memory)
        for e in range(epochs):
            # Resets the environment per epoch
            state_ = self.env.reset()
            # A flag to check if an episode has ended
            done = False
            # Stores the score per episode
            score = 0
            # Maximum number of allowed moves per episode
            maxMoves = 200
            # Stores the information per move in an episode
            iterations = []            
            # Continue the episode until it ends or the maximum number of episodes has expired
            while not done and maxMoves > 0:
                # Decrement the number of allowed moves
                maxMoves -= 1
                # Calculate the probs. for the actions for a given state
                policy = self.actor(torch.from_numpy(
                    state_).float())
                # Choose an action based on their probs.
                action = np.random.choice(
                    self.numActionSpace, p=policy.detach().numpy())
                # Executes an action to the environment
                nextState_, reward, done, _ = self.env.step(action)
                # Updates the rewards for the episode
                score += reward
                # Calculates the value of the current state
                value = self.critic(torch.from_numpy(state_).float())
                # Calculate the value of the next state
                nextValue = torch.Tensor([0.0]) if done else self.critic(
                    torch.from_numpy(nextState_).float())
                
                # Add the experience to the buffer
                replay.append((state_, reward, nextValue.item()))
                # Add the episode step info
                iterations.append(
                    (state_, action, reward, nextValue.item() - value.item())) 
                # Assign next state as current state
                state_ = nextState_

            # After each episode update the actor model
            # Policy Loss
            iterations_ = np.array(iterations)  
            # Calculate the advantage
            advantage = torch.Tensor(list(iterations_[:, 2])).float() + torch.pow(gamma, torch.arange(
                len(iterations_)).flip(dims=(0, )).float()) * torch.Tensor(list(iterations_[:, 3])).float()

            # Store the state info as a batch of states
            stateBatch = torch.stack([torch.from_numpy(s).float()
                                      for s in iterations_[:, 0]])
            # Store the action info as a batch of actions    
            actionBatch = torch.Tensor(list(iterations_[:, 1]))
            # Feed the state batch to the actor model to calculate the probs of actions for each state in the batch
            policy = self.actor(stateBatch)
            # Gets the probs of actions actually performed for each state
            probs = policy.gather(
                dim=1, index=actionBatch.long().unsqueeze(dim=1)).squeeze()

            # Policy Loss
            actorLoss = self.actorLossFun(probs, advantage)
            #Backpropagate policy
            self.actorOptim.zero_grad()
            actorLoss.backward()
            self.actorOptim.step()

            # Update the value function if the size of the replay buffer is larger than the specified batch size
            if (len(replay) > batch):
                # Select a set of random indices to be chosen from the replay buffer 
                indices = np.random.choice(len(replay), size=batch)
                # Extract the experiences from the replay buffer
                replay_ = np.array(replay)[indices, :]  
                # Create a state batch with the excted experiences
                stateBatch = torch.stack([torch.from_numpy(s).float()
                                          for s in replay_[:, 0]])
                # Calculate the value for the extracted states                                        
                value = self.critic(stateBatch)                                                
                
                # Value Loss
                criticLoss = self.criticLossFun(
                    value, torch.Tensor(list(replay_[
                        :, 1] + gamma * replay_[:, 2])).float())
                                
                #Backpropagate value
                self.criticOptim.zero_grad()
                criticLoss.backward()
                self.criticOptim.step()            
                
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
    agent = A2CBatch(env)
    # Train the agent
    info = agent.train(1000)
    # Convert to an numpy array with epochs at axis=0 and scores at axis=1
    info_ = np.array(list(info.items()))
    # Plot the scores
    agent.plot(info_)
    # Run a test
    agent.test()
