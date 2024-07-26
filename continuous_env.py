from copy import deepcopy
import numpy as np
import math

class ContinuousGridWorld:
    def __init__(self):
        self.world = [[0,0],[10,10]]            # [[begin_x,begin_y],[end_x,end_y]]
        self.initial_pos = [1,0]                # [x,y]
        self.dist = 1                           # Agent only allowed to move 1 unit distance
        self.goal = [[8.5,0],[9.5,1]]           # [[begin_x,begin_y],[end_x,end_y]]
        self.midgoal = [[4.5,1.5],[5.5,2.5]]
        self.trap = [[4,0],[6,1]]               # [[begin_x,begin_y],[end_x,end_y]]
        self.trig_tolerance = 1e-15             # A small tolerance for considering the result of trigonometric calculations as 0
        self.num_actions = 36                   # The agent has 36 possible actions in 10 degree increaments
        self.steps = 0
        self.dt = 0.5
        self.n_points = 50                      #To be tuned (75  might be to much) (20 is acceptable) (30 the best so far)
        self.max_z = self.trap[1][1] + 2
        self.w = [1,-100,2,25,15,50,-25]              #weights of the rewarding function
        self.r = [0,0,0,0,0,0,0]                    #raw value of the rewarding function (movement, trap, dy, goal, midpoint, dz, dt)

    # Resets agent back to starting position
    def reset(self):
        self.is_done = False
        self.cur_state = deepcopy(self.initial_pos)
        return self.cur_state

    # Returns the number of dimensions of the grid
    def _get_state_dim(self):
        return len(self.world[0])

    # The agent has 36 possible actions in 10 degree increaments
    def _get_action_dim(self):
        return self.num_actions

    def get_min_distance_from_trap(self, state):
        min = 10000
        for i in range(self.trap[0][0],self.trap[1][0]*10):            #i want a cent as tollerance
            for j in range(self.trap[0][1],self.trap[1][1]*10):
                temp = ((state[0]-(i/10))**2 + (state[1]-(j/10))**2)**0.5
                if (min > temp):
                    min = temp
        return min

    # Transition function to calculate rewards and position of next state
    def transition(self, state, action):
        step_limit = 150
        reward = 0
        if self.is_done:
            return 0, state, True

        # Agent can only move 1 unit distance. x = 1*cos(theta), y = 1*sin(theta)
        self.steps += 1
        x = math.cos(action)
        y = math.sin(action)
        if abs(x) < self.trig_tolerance:
            x = 0
        if abs(y) < self.trig_tolerance:
            y = 0
        
        # Assuming moving 0 degrees always brings you north, clip the agent's coordinates so that it doesn't move out of the grid
        next_state_x = state[0] + (self.dist * x)
        next_state_x = np.clip(next_state_x, self.world[0][0], self.world[1][0])
        next_state_y = state[1] + (self.dist * y)
        next_state_y = np.clip(next_state_y, self.world[0][1], self.world[1][1])
        next_state = [next_state_x, next_state_y]

        # Define rewards
        # If goal is reached
        #print(f"num of steps: {self.steps}")
        #self.w = self.w/np.sum(self.w)
        if (self.steps == step_limit):
            self.is_done = True
        if(state[0] > next_state[0]):
            self.r[2] = -20
        else:
            self.r[2] = 0
        if (next_state[1] > self.max_z):
            self.r[5] = -50
        else:
            self.r[5] = 0
        if (self.goal[0][0] <= next_state[0] <= self.goal[1][0]) and (self.goal[0][1] <= next_state[1] <= self.goal[1][1]):
            self.r[3] = 100
            self.is_done = True
        else:
            self.r[3] = 0
        # If agent is in the trap
        if (self.trap[0][0] <= next_state[0] <= self.trap[1][0]) and (self.trap[0][1] <= next_state[1] <= self.trap[1][1]):
            self.r[1] = 50
        else:
            self.r[1] = 0
        # If agent is not in the goal or trap
        if (self.midgoal[0][0] <= next_state[0] <= self.midgoal[1][0]) and (self.midgoal[0][1] <= next_state[1] <= self.midgoal[1][1]):
            self.r[4] = 50
        else:
            self.r[4] = 0
        if self.steps <= self.n_points:
            self.r[0] = self.steps/self.n_points
        else:
            self.r[0] = -self.steps/self.n_points
        if (self.get_min_distance_from_trap(next_state) <= self.dt):
            self.r[6] = 20
        else:
            self.r[6] = 0
        reward = (self.w[0]*self.r[0])+(self.w[1] * self.r[1])+(self.w[2] * self.r[2])+(self.w[3] * self.r[3])+(self.w[4] * self.r[4])+(self.w[5] * self.r[5])+(self.w[6]*self.r[6])

        return reward, next_state, self.is_done