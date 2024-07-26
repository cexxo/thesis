#baseline code available at https://medium.com/mitb-for-all/reinforcement-learning-implementing-td-%CE%BB-with-function-approximation-9b5f9f640aa1

import matplotlib.pyplot as plt
import torch

from net import TemporalDifference
from continuous_env import ContinuousGridWorld

print(torch.cuda.is_available())


# Set the environment
env = ContinuousGridWorld()
training_episodes = 10_000
agents = {}
print(f"goal_x: {env.goal[0][0]}    goal_y: {env.goal[0][1]}")


agent_QLearning = TemporalDifference(env, alpha=0.001, gamma=0.99, epsilon=0.1, lambd=0)
agents['Q-Learning'] = (agent_QLearning, False)
agent_QLearning.train(num_episodes=training_episodes, on_policy=False)

"""agent_SARSA = TemporalDifference(env, alpha=0.005, gamma=0.99, epsilon=0.1, lambd=0)            #was epsilon=0.1, alpha=0.001
agents['SARSA'] = (agent_SARSA, True)
agent_SARSA.train(num_episodes=training_episodes)"""


def play_sample_episode(agent, env):
    #print("play sample episode")
    env.steps = 0
    state = env.reset()
    done = False
    path = [] # Keeps track of the states visited in the episode
    step, steps_trapped, total_reward = 0, 0, 0

    while not done:
        #print(f"executing episode, step n: {step}      done: {done}")
        action = agent.epsilon_greedy_policy(state)
        path.append(state)
        reward, next_state, done = env.transition(state, action)
        if reward == -50:
            steps_trapped += 1
        total_reward += reward
        state = next_state
        step += 1

    # Append the terminal state
    path.append(state)

    return step, path, steps_trapped, total_reward

test_episodes = 1_000
agents_results = {} # Store results in this format {(agent, episode): {'path':path, 'step':step, 'returns':returns}}

for key, value in agents.items():
    name = key
    agent = value[0]
    on_policy = value[1]
    # Create holders for results for each agent
    episodes, steps, paths, returns, trapped = [], [], [], [], []
    param = f'α={agent.alpha}, ɣ={agent.gamma}, ε={agent.epsilon}, λ={agent.lambd}, on_policy={on_policy}'
    for episode in range(test_episodes):
        #print(f"num test episode: {episode}")
        step, path, steps_trapped, total_reward = play_sample_episode(agent, env)
        episodes.append(episode)
        steps.append(step)
        paths.append(path)
        returns.append(total_reward)
        trapped.append(steps_trapped)

        agents_results[(name, episode)] = {'path':path, 'step':step, 'returns':returns[episode], 'params':param}

def plot_path(agent_name, episode_num=0):
    path = agents_results[agent_name, episode_num]['path']
    steps = agents_results[agent_name, episode_num]['step']
    episode_return = agents_results[agent_name, episode_num]['returns']
    params = agents_results[agent_name, episode_num]['params']

    width = 90
    
    x = [state[0] for state in path]
    y = [state[1] for state in path]
    
    # For plotting arrows to show each step the agent takes
    u, v = [], []
    for i in range(1,len(x)):
        u.append(x[i] - x[i-1])
        v.append(y[i] - y[i-1])
    u.append(0) # Add extra 0 to match the number of positions
    v.append(0) # Add extra 0 to match the number of positions

    # Plot world
    fig, ax = plt.subplots()
    ax.set_aspect('equal') # Sets a square grid
    ax.set_title(f"TemporalDifference({params})")
    ax.set_xlim(env.world[0][0], env.world[1][0])
    ax.set_ylim(env.world[0][1], env.world[1][1])
    # Plot goal
    g_width = env.goal[1][0] - env.goal[0][0]
    g_height = env.goal[1][1] - env.goal[0][1]
    ax.add_patch(plt.Rectangle((env.goal[0][0], env.goal[0][1]), g_width, g_height, color = 'yellow'))
    ax.annotate('G', (env.goal[0][0] + 0.5, env.goal[0][1] + 0.5), color='black', size=14, ha='center', va='center')
    # Plot trap
    t_width = env.trap[1][0] - env.trap[0][0]
    t_height = env.trap[1][1] - env.trap[0][1]
    ax.add_patch(plt.Rectangle((env.trap[0][0], env.trap[0][1]), t_width, t_height, color = 'grey'))
    ax.annotate('T', (env.trap[0][0] + 0.5, env.trap[0][1] + 0.5), size=14, ha='center', va='center')
    # Plot start
    ax.plot(env.initial_pos[0],env.initial_pos[1],'ro')
    ax.annotate('S', (env.initial_pos[0],env.initial_pos[1]), size=14, ha='center', va='center')
    #Plot midpoint
    m_width = env.midgoal[1][0] - env.midgoal[0][0]
    m_height = env.midgoal[1][1] - env.midgoal[0][1]
    ax.add_patch(plt.Rectangle((env.midgoal[0][0], env.midgoal[0][1]), m_width, m_height, color = 'green'))
    ax.annotate('M', (env.midgoal[0][0] + 0.5, env.midgoal[0][1] + 0.5), color='black', size=14, ha='center', va='center')
    # Plot steps: arrows at coordinates (x, y) with directions (u, v)
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)

    #plt.show()
    plt.savefig(f"./results/{agent_name}_{episode_num}.png")

for i in range(test_episodes):
    for agent_name in agents.keys():
        plot_path(agent_name, i)