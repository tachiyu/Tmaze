import numpy as np
import utils
import random

class TmazeWorld(object):
    def __init__(self, size=80, block_pattern="empty", 
                 verbose=False, obs_mode="onehot"):
        self.verbose = verbose
        self.action_size = 2
        self.obs_mode = obs_mode
        self.state_size = size
        self.goal_pos = [28, 78]
        self.end_pos = [29, 39, 69, 79]
        self.start_pos = [0, 40]
        self.junc_pos = [19, 59]
        self.agent_pos = []
        self.obs_size = None
        self.observations = None
        if obs_mode == "onehot":
            self.obs_size = self.state_size
            self.goal_size = self.state_size
        if obs_mode == "twohot":
            self.obs_size = self.grid_size * 2
            self.goal_size = self.grid_size * 2
        if obs_mode == "geometric":
            self.obs_size = 2
            self.goal_size = 2
        if obs_mode == "index":
            self.obs_size = 1
            self.goal_size = 1
            
    def reset(self, agent_pos=None):
        self.done = False
        if agent_pos != None:
            self.agent_pos = agent_pos
        else:
            self.agent_pos = random.choice(self.start_pos)
        
    def state_to_coordinate(self,state):
        i = state
        if i < 20:
            return [i+1, 11]
        elif i < 30:
            return [20, 11-(i-20)-1]
        elif i < 40:
            return [20, 11+(i-30)+1]
        elif i < 60:
            return [(i-40)+1, 33]
        elif i < 70:
            return [20, 33-(i-60)-1]
        elif i < 80:
            return [20, 33+(i-70)+1]
        :/\
            
            
            
        
    @property
    def grid(self):
        grid = np.zeros([22, 45, 3])
        grid[self.state_to_coordinate(self.agent_pos)[0], self.state_to_coordinate(self.agent_pos)[1], 0] = 1
        for goal in self.goal_pos:
            grid[self.state_to_coordinate(goal)[0], self.state_to_coordinate(goal)[1], 1] = 1
        for i in range(80):
            grid[self.state_to_coordinate(i)[0], self.state_to_coordinate(i)[1], 2] = 1
        return grid
    
    def move_agent(self, direction):
        if direction == "start":
            new_pos = random.choice([0,40])
        else:
            new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)
            
    def simulate(self, action):
        agent_old_pos = self.agent_pos
        reward = self.step(action)
        state = self.state
        self.agent_pos = agent_old_pos
        return state
        
        
    @property
    def observation(self):
        return self.agent_pos

    @property
    def goal(self):
        if self.obs_mode == "onehot":
            return utils.onehot(self.goal_pos[0] * self.grid_size + self.goal_pos[1], self.state_size)
        if self.obs_mode == "twohot":
            return self.twohot(self.goal_pos, self.grid_size)
        if self.obs_mode == "geometric":
            return (2 * np.array(self.goal_pos) / (self.grid_size-1)) - 1 
        if self.obs_mode == "visual":
            return env.grid
        if self.obs_mode == "index":
            return self.goal_pos[0] * self.grid_size + self.goal_pos[1]
        
    @property
    def all_positions(self):
        all_positions = self.blocks + [self.goal_pos] + [self.agent_pos]
        return all_positions
    
    def state_to_obs(self, state):
        if self.obs_mode == "onehot":
            point = self.state_to_point(state)
            return utils.onehot(point[0] * self.grid_size + point[1], self.state_size)
        if self.obs_mode == "twohot":
            point = self.state_to_point(state)
            return self.twohot(point, self.grid_size)
        if self.obs_mode == "geometric":
            point = self.state_to_point(state)
            return (2 * np.array(point) / (self.grid_size-1)) - 1 
        if self.obs_mode == "visual":
            return self.state_to_grid(state)
        if self.obs_mode == "index":
            return state
            
    def step(self, action):
        # 0 - Left
        # 1 - Right
        #if agent in t-junction, action can be left or right
        if self.agent_pos in self.junc_pos:
            if action == 0:
                self.agent_pos += 1
            if action == 1:
                self.agent_pos += 11
        #if agent in end of arm, agent moves to random start.
        elif self.agent_pos in self.end_pos:
            self.agent_pos = random.choice(self.start_pos)
        else:
            self.agent_pos += 1
        if self.agent_pos in self.goal_pos:
            return 1.0
        else:
            return 0.0

    def state_to_goal(self, state):
        return self.state_to_obs(state)
