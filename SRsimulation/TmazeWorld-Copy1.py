import numpy as np
import utils
import random

class TmazeWorld(object):
    def __init__(self, size=25, block_pattern="empty", 
                 verbose=False, obs_mode="onehot"):
        self.verbose = verbose
        self.grid_size = size
        self.action_size = 2
        self.obs_mode = obs_mode
        self.state_size = size*size
        self.blocks = self.make_blocks(block_pattern)
        self.goal_pos = []
        self.end_pos = []
        self.start_pos = []
        self.junc_pos = []
        self.agent_pos = []
        self.obs_size = None
        self.done = None
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
            
    def reset(self, goal_pos=None, end_pos=None, start_pos=None, junc_pos=None, agent_pos=None):
        self.done = False
        if goal_pos != None:
            self.goal_pos = goal_pos
        else:
            self.goal_pos = self.get_free_spot()
        if end_pos != None:
            self.end_pos = end_pos
        else:
            self.end_pos = self.get_free_spot()
        if start_pos != None:
            self.start_pos = start_pos
        else:
            self.start_pos = self.get_free_spot()       
        if junc_pos != None:
            self.junc_pos = junc_pos
        else:
            self.junc_pos = self.get_free_spot()   
        if agent_pos != None:
            self.agent_pos = agent_pos
        else:
            self.agent_pos = self.get_free_spot()
        
    def get_free_spot(self):
        free = False
        possible_x = np.arange(0, self.grid_size)
        possible_y = np.arange(0, self.grid_size)
        while not free:
            try_x = np.random.choice(possible_x, replace=False)
            try_y = np.random.choice(possible_y, replace=False)
            try_position = [try_x, try_y]
            if try_position not in self.all_positions:
                return try_position
        
    def make_blocks(self, pattern):
        if pattern == "four_rooms":
            mid = int(self.grid_size // 2)
            earl_mid = int(mid // 2)
            late_mid = mid+earl_mid + 1
            blocks_a = [[mid,i] for i in range(self.grid_size)]
            blocks_b = [[i,mid] for i in range(self.grid_size)]
            blocks = blocks_a + blocks_b
            self.bottlenecks = [[mid,earl_mid],[mid,late_mid],[earl_mid,mid],[late_mid,mid]]
            for bottleneck in self.bottlenecks:
                blocks.remove(bottleneck)
            return blocks
        if pattern == "empty":
            self.bottlenecks = []
            return []
        if pattern == "random":
            blocks = []
            for i in range(self.state_size // 10):
                blocks.append([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            self.bottlenecks = []
            return blocks
        if pattern == "two_rooms":
            mid = int(self.grid_size // 2)
            blocks = [[mid,i] for i in range(self.grid_size)]
            blocks.remove([mid,mid])
            self.bottlenecks = [[mid,mid]]
            return blocks
        if pattern == "tmaze":
            blocks = []
            for i in range(self.grid_size):
                if i < 3:
                    blocks += [[i,j] for j in range(self.grid_size)]
                elif i == 3:
                    blocks += [[i,j] for j in (0,12,24)]
                elif i < 23:
                    blocks += [[i,j] for j in range(self.grid_size) if j not in (6,18)]
                else:
                    blocks += [[i,j] for j in range(self.grid_size)]
            self.bottlenecks = []
            return blocks
        
    @property
    def grid(self):
        grid = np.zeros([self.grid_size, self.grid_size, 3])
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        for goal in self.goal_pos:
            grid[goal[0], goal[1], 1] = 1
        for block in self.blocks:
            grid[block[0], block[1], 2] = 1
        return grid
    
    def move_agent(self, direction):
        if direction == "start":
            new_pos = random.choice([[22,6], [22,18]])
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
        
    def check_target(self, target):
        x_check = target[0] > -1 and target[0] < self.grid_size
        y_check = target[1] > -1 and target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        if x_check and y_check and block_check:
            return True
        else:
            return False
        
    @property
    def observation(self):
        if self.obs_mode == "onehot":
            return utils.onehot(self.agent_pos[0] * self.grid_size + self.agent_pos[1], self.state_size)
        if self.obs_mode == "twohot":
            return self.twohot(self.agent_pos, self.grid_size)
        if self.obs_mode == "geometric":
            return (2 * np.array(self.agent_pos) / (self.grid_size-1)) - 1 
        if self.obs_mode == "visual":
            return env.grid
        if self.obs_mode == "index":
            return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

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
    
    def state_to_grid(self, state):
        vec_state = np.zeros([self.state_size])
        vec_state[state] = 1
        vec_state = np.reshape(vec_state, [self.grid_size, self.grid_size])
        return vec_state
    
    def state_to_goal(self, state):
        return utils.onehot(state, self.state_size)
    
    def state_to_point(self, state):
        a = self.state_to_grid(state)
        b = np.where(a==1)
        c = [b[0][0],b[1][0]]
        return c
    
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
        # 2 - Stay
        move_array = np.array([0,0])
        if action == 2:
            move_array = np.array([0,0])
            self.move_agent(move_array)
            return 0.0
        else:
            #if agent in t-junction, action can be left or right
            if self.agent_pos in self.junc_pos:
                if action == 0:
                    move_array = np.array([0,-1])
                if action == 1:
                    move_array = np.array([0,1])   
                self.move_agent(move_array)
            #if agent in end of arm, agent moves to random start.
            elif self.agent_pos in self.end_pos:
                self.move_agent("start")
            else:
                #if agent in left arm
                if 0 < self.agent_pos[1] < 6 or 12 < self.agent_pos[1] < 18:
                    move_array = np.array([0,-1])
                #if agent in right arm
                elif 6 < self.agent_pos[1] < 12 or 18 < self.agent_pos[1] < 24:
                    move_array = np.array([0,1])
                #if agent in stem
                elif 3 < self.agent_pos[0] < 23:
                    move_array = np.array([-1,0])
                self.move_agent(move_array)
            if self.agent_pos in self.goal_pos:
                return 1.0
            else:
                return 0.0

    def state_to_goal(self, state):
        return self.state_to_obs(state)
