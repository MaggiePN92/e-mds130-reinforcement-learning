import torch
import numpy as np
from matplotlib import pyplot as plt

class A2C:
    def __init__(self, env):
        # Set the seed values
        torch.seed = 42
        torch.manual_seed(torch.seed)
        np.random.seed(torch.seed)
        
        # Setting up the environment
        self.env = env
        # Set the environment seed
        self.env.seed = torch.seed
        # Store the size of the action and observation space
        self.numStateSpace = self.env.observation_space.shape[0]
        self.numActionSpace = self.env.action_space.n

        # Create a model for the actor - policy
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.numStateSpace, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, self.numActionSpace),
            torch.nn.Softmax(dim=-1)
        )

        # Create a model for the critic - value
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.numStateSpace, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
            # torch.nn.Tanh()
        )
        # Optimizer for the actor
        self.actorOptim = torch.optim.Adam(self.actor.parameters(), lr=0.001)        
        # Optimizer for the critic
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        # Loss function for the critic
        self.criticLossFun = torch.nn.MSELoss()

    # Loss function for the actor. We may have defined it as a lambda function inside the constructor as well, 
    # however, doing so will create issues with multiprocessing if this class is inherited by others
    def actorLossFun(self, probs, advantage): 
        return - 1 * torch.log(probs) * advantage

    # The train function. Will be overritten by classes that inherit this class
    def train(self):
        pass

    # A test function that tests using the trained actor model on a new instance of the environment
    def test(self, render=True):
        # Reset the environment for testing
        state_ = self.env.reset()
        # Flag to determine if an episode has ended
        done = False
        # Maximum number of moves allowed while testing
        maxMoves = 500
        # Stores the total score received during an episode
        score = 0
        # Continues an episode until it ends or the maximim number of allowed moves has expired
        while not done and maxMoves > 0:            
            # Decrement the maximim number of allowed moves per play in an episode
            maxMoves -= 1
            # If render is true, renders the game to screen
            if render:
                self.env.render()
            # Calculates the probs. for the actions given a state
            policy = self.actor(torch.from_numpy(
                state_).float())
            # Chooses an action based on their probs.
            action = np.random.choice(
                len(policy), p=policy.detach().numpy())
            # Executes an action in the environment
            state_, reward, done, _ = self.env.step(action)
            # Stores the reward
            score += reward
        # Print the rewards received         
        print('reward: {}'.format(score))
    
    # A function to plot the scores received over a number of epochs
    def plot(self, info_):
        # Sort the scores by epoch
        info_.sort(axis=0)
        # Extract the epochs and respective scores
        x, y = info_[:, 0], info_[:, 1]
        
        plt.plot(x, y)
        plt.title('Scores')
        plt.xlabel('episode')
        plt.ylabel('score')        
        plt.show()
