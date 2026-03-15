Simple Summary from what I learned and built

Ver 1.0
Grid Size: 5 x 5
Learning Rate: 0.1 
Discount Factor: 0.99
Episodes: 1000
Epsilon Decay: 0.995 to 0.01 (explores less ove time)

Tool Used: Python, Numpy, Matplotlib (No deep learning frameworks used, kept it simple)

I built a simple 5x5 grid world where a robot learns to navigate from the top-left corner to the bottom-right corner using Q-learning.

In the beginning, the robot has no idea what is on the grid. It will exlore randomly at first and collecting rewards and penalities. After each move, it updates a Q-table with how good that move was by the rewards and penalties. Penalities include blocking robot from just sitting in the same spot. I set the limit to 1000 episodes; it learns the best/optimal path.

Early episodes: rewards can get as low as -500. I see that as random warnering. By episodes around 200: rewards stabilize near 92. The learned path started to match the shortest BFS path and match almost completely exactly 8 steps. Rewards can never reach 100 because each step costs -1 which lead to 100 rewards - 8 steps (each steps cost -1) = 92 rewards. 

Ver 2.0
I made some changes. 
Every step, there is a 20% chance of taking a random step toward any directions. 
And I updated the grid to be random size between 5x5 to 10x10.
Randomize the start and goal, also, how many walls there are between 10 to 15% of the remaining amount of cells. 