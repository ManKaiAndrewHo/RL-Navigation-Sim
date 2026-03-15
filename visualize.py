#Grid World Q-learning visualization
#Train an agent, extracts paths, and plots results
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from grid_world import GridWorld, train, get_shortest_path, get_learned_path

#run everything
env = GridWorld()
q_table, rewards = train(env)
learned_path = get_learned_path(env, q_table)
shortest_path = get_shortest_path(env)

print("Grid Size: ", env.grid_size)
print("Start: ", env.start)
print("Goal: ", env.goal)
print("Walls: ", len(env.wall))
print("Learned Path: ", len(learned_path) - 1)
print("Shortest Path: ", len(shortest_path) - 1)

#Plot 1: grid with paths
def plot_grid(env, learned, shortest):
    fig, ax = plt.subplots(figsize = (6, 6)) #fig is the canvas frame, ax is the canvas itself

    for x in range(env.grid_size):
        for y in range(env.grid_size):
            color = "white"
            if (x, y) in env.wall:
                color = "black"
            elif (x, y) == env.start:
                color = "green"
            elif (x, y) == env.goal:
                color = "gold"
            #(y, 4 - x) = bottom left corner of the rectangle
            #4 - x because matplotlib draws from bottom to top but our grid is from top to bottom
            #two 1s are the width and height of the cell
            rect = patches.Rectangle((y, env.grid_size - 1 - x), 1, 1, linewidth = 1, edgecolor = "gray", facecolor = color)
            ax.add_patch(rect)
            
    #draw shortest path in blue
    for (x, y) in shortest:

        ax.plot(y + 0.5, env.grid_size - 1 - x + 0.5, "bs", markersize = 10, alpha = 0.4)

    #draw learned path in red
    for (x, y) in learned:
        ax.plot(y + 0.5, env.grid_size - 1 - x + 0.5, "r.", markersize = 8)

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_title("Learned Path (red) vs Shortest Path (blue)")
    ax.axis("on")
    plt.tight_layout()
    plt.show()
    
#Plot 2: Reward convergence
def plot_rewards(rewards):
    smoothed = np.convolve(rewards, np.ones(50) / 50, mode = "valid")
    plt.figure(figsize = (8, 4))
    plt.plot(rewards, alpha = 0.3, color = "blue", label = "Raw Rewards")
    plt.plot(smoothed, color = "red", label = "Smoothed (50 ep avg)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_grid(env, learned_path, shortest_path)
plot_rewards(rewards)