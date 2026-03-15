import numpy as np
import random
from collections import deque

class GridWorld:
    def __init__(self):
        # self.grid_size = 5 #5x5 grid
        self.grid_size = random.randint(5, 10) #random grid size between 5 to 10
        # self.start = (0,0) #top-left corner
        # self.goal = (4,4) #bottom-right corner
        # self.wall = [(1,1), (2,2), (3,1)] #blocked cells
        self.start, self.goal, self.wall = self.generate_random_grid() #generate random start, goal and walls
        self.agent = self.start #robot starts at start
        
    def generate_random_grid(self):
        while True:
            #all start and goal are not the same spots and random
            all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)] 
            start, goal = random.sample(all_cells, 2)
            
            #10-15% of remaining cells are walls
            remaining = [cell for cell in all_cells if cell != start and cell != goal]
            wall_count = int(len(remaining) * random.uniform(0.10, 0.15))
            walls = set(random.sample(remaining, wall_count))
            
            #only accept if BFS found a solution
            if self.has_solution(start, goal, walls):
                return start, goal, walls
            
    def has_solution(self, start, goal, walls):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            x, y = current
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
        
    def reset(self):
        self.agent = self.start #put robot back at start
        return self.agent
    
    def step(self, action):
        if np.random.random() < 0.2:
            action = np.random.randint(4) #20% chance to take random action
        x, y = self.agent
        
        #0 = up, 1 = down, 2 = left, 3 = right
        if action == 0: 
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1
            
        #hit a wall or out of bounds
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or (x,y) in self.wall:
            return self.agent, -10, False #penalty for staying in place
        
        self.agent = (x,y)
        
        if self.agent == self.goal:
            return self.agent, 100, True #reached goal
        
        return self.agent, -1, False #normal step
    
#Q-learning setup
#Train the agent using Q-learning and returns the Q-table and rewards
def train(env, episodes = 2000):
    # env = GridWorld()

    # Q-table: one row per cell, one column per action
    # q_table = np.zeros((5, 5, 4))  # (x, y, action)
    q_table = np.zeros((env.grid_size, env.grid_size, 4)) #dynamic size now

    learning_rate = 0.1
    discount = 0.99
    epsilon = 1.0          # how often to explore randomly
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            x, y = state

            # explore randomly OR use Q-table
            if np.random.random() < epsilon:
                action = np.random.randint(4)   # random action
            else:
                action = np.argmax(q_table[x, y])  # best known action

            next_state, reward, done = env.step(action)
            nx, ny = next_state

            # update Q-table
            old_value = q_table[x, y, action]
            next_max = np.max(q_table[nx, ny])
            q_table[x, y, action] = old_value + learning_rate * (reward + discount * next_max - old_value)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)  # explore less over time

    return q_table, rewards_per_episode

# q_table, rewards = train()
# print("Training done!")
# print("Best action at start (0,0):", np.argmax(q_table[0, 0]))  # should be 1 or 3

#Extract the learned path the agent takes using the learned Q-table
def get_learned_path(env, q_table):
    # env = GridWorld()
    state = env.reset()
    path = [state]
    done = False
    steps = 0
    
    # while not done and steps < 50: #50 steps avoid infinite loops
    while not done and steps < env.grid_size * env.grid_size:
        x, y = state
        action = np.argmax(q_table[x, y])  #always use best action
        state, reward, done = env.step(action)
        path.append(state)
        steps += 1
        
    return path

# learned_path = get_learned_path(q_table)
# print("Learned path: ", learned_path)

#Find the shortest possible path using BFS (Breadth-First Search)
def get_shortest_path(env):
    # env = GridWorld()
    # queue = deque()
    # queue.append([(0, 0)]) #start with a path containing just the start (0, 0)
    # visited = set()
    # visited.add((0, 0))
    # start = (0, 0)
    # goal = (4, 4)
    queue = deque([[env.start]])
    visited = set([env.start])
    
    while queue:
        path = queue.popleft() 
        # current = path[-1] 
        x, y = path[-1]
        
        # if current == goal:
        #     return path
        if (x, y) == env.goal:
            return path
        
        # x, y = current 
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        
        for nx, ny in neighbors:
            if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size and (nx, ny) not in env.wall and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(path + [(nx, ny)])
        
        # for action in range(4):
        #     if action == 0:
        #         nx, ny = x - 1, y
        #     elif action == 1:
        #         nx, ny = x + 1, y
        #     elif action == 2:
        #         nx, ny = x, y - 1
        #     elif action == 3:
        #         nx, ny = x, y + 1
                
        # if (0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in env.wall and (nx, ny) not in visited):
        #     visited.add((nx, ny))
        #     queue.append(path + [(nx, ny)]) 
    
    return None

# shortest_path = get_shortest_path()
# print("Shortest Path: ", shortest_path)
# print("Learned Steps: ", len(learned_path) - 1)
# print("Shortest Steps: ", len(shortest_path) - 1)