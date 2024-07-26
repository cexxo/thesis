import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
from collections import  deque

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Hidden layers. In this case, we will use 3 hidden layers.
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    # Feed forward
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class TemporalDifference:
    def __init__(self, Env, alpha=0.001, gamma=0.9, epsilon=0.1, lambd=0.9, batch_size=32):
        self.Env = Env
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.name = f"TemporalDifference(α = {self.alpha}, ɣ = {self.gamma}, ε = {self.epsilon}, λ = {self.lambd})"
        
        self.state_dim = self.Env._get_state_dim()      # Get dimensions of the state (in this case, state_dim=2 because the environment is 2D)
        self.action_dim = self.Env._get_action_dim()    # Get number of actions

        # Create two neural networks: Q_main and Q_target
        self.Q_main = Net(self.state_dim, self.action_dim)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)

    # Action selection strategy
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)    # Explore by taking a random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.Q_main(state_tensor)).item()   # Exploit by taking the best action

    # Reset Q_main and Q_target
    def reset(self):
        self.Q_main = Net(self.state_dim, self.action_dim)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)

    # Update Q_target (soft update)
    def _soft_update_Qtarget(self, tau=0.01):
        with torch.no_grad():
            for target_param, param in zip(self.Q_target.parameters(), self.Q_main.parameters()):
                target_param += tau * (param - target_param)

    # Update Q_main's weights, similar to backpropagation
    def _update_Qmain_weights(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_Qtarget()

    # Reset the episode if it gets too long
    def reset_episode(self):
        state = self.Env.reset()
        done = False
        step = 0
        episode_return = 0
        trace_dict = {} # For storing eligibility traces {(state's x coordinate, state's y coordinate, action): trace value}
    
        return state, done, step, episode_return, trace_dict
    
    # Train the agent
    def train(self, num_episodes, on_policy=True):
        memory = deque(maxlen=10_000)       # Memory of all episodes performed
        step_limit = 150                  # To limit the number of steps, so that we can kill the episode if the number of steps go out of hand. 500 is to less, 1000 to much noise

        # Iterate through episodes
        for episode in tqdm(range(num_episodes), desc="Episodes", position=0, leave=True):
            # Initialise by resetting
            state, done, step, episode_return, trace_dict = self.reset_episode()
            action = self.epsilon_greedy_policy(state)

            while not done and step < step_limit:
                reward, next_state, done = self.Env.transition(state, action)
                next_action = self.epsilon_greedy_policy(next_state)

                # Increment trace for current state
                trace_key = (state[0],state[1],action)
                if trace_key not in trace_dict:
                    trace_dict[trace_key] = 0
                trace_dict[trace_key] += 1
                trace = list(trace_dict.values())

                # Store in memory buffer
                memory.append((episode, state, action, reward, next_state, next_action, done, trace))

                # Decay trace for past visited states
                trace_dict[trace_key] = (self.gamma**step) * (self.lambd**step)

                # Update state, action, episode_return, step
                state, action = next_state, next_action
                episode_return += reward
                step += 1
                
                # Discard the episode if the agent reaches the step_limit but is still unable to reach the goal (i.e. the agent is stuck)
                """if step >= step_limit and not done:
                    print('Episode reset, agent stuck')
                    state, done, step, episode_return, trace_dict = self.reset_episode()
                    action = self.epsilon_greedy_policy(state)
                    memory = [tup for tup in memory if tup[0] != episode] # remove the bad episode from memory"""
                if step >= step_limit:
                    done = True
                # Once there are sufficient samples in memory, randomly sample a batch to update Q network
                if len(memory) >= self.batch_size:
                    batch = random.choices(memory, k = self.batch_size)
                    self.replay(batch, on_policy)

    # Replaying the batch of episodes to train the neural networks
    def replay(self, batch, on_policy):
        # Unpack batch and convert to required types
        episodes, states, actions, rewards, next_states, next_actions, dones, traces = zip(*batch)
        states = torch.tensor(states).to(torch.float32)
        actions = torch.tensor(actions).to(torch.int64)
        rewards = torch.tensor(rewards).to(torch.float32)
        next_states = torch.tensor(next_states).to(torch.float32)
        next_actions = torch.tensor(next_actions).to(torch.int64)
        dones = torch.tensor(dones).to(torch.int16)

        # Get next_q from Q_target
        if on_policy==True:
            # If acting on policy, get next_q from the next action taken
            next_q = self.Q_target(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            # If NOT acting on policy, get next_q from the best possible next action
            next_q = self.Q_target(next_states).max(1)[0]

        targets = rewards + (self.gamma * next_q * (1 - dones))

        # Get current_q from Q_main
        current_q = self.Q_main(states)                                     # q values across all possible actions
        current_q = current_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Pick the q value for the corresponding action taken

        # Explode elibility traces to get the trace for that state
        traces = [torch.tensor(trace).to(torch.float32) for trace in traces]
        current_q = torch.cat([torch.mul(trace, q) for trace, q in zip(traces, current_q)])
        targets = torch.cat([torch.mul(trace, target) for trace, target in zip(traces, targets)])


        # Loss function: q_pred - q_target (where q_target = reward + gamma*next_q)
        loss = nn.MSELoss()(current_q, targets)

        # Update Q_main and perform soft update of Q_target
        # The soft update of Q_target is in the ._update_Qmain_weights function
        self._update_Qmain_weights(loss)