### The Story
The world is a simplified version of tetris. The agent is trying to find the best column to place a block in order to have a row be filled. In our example the board is 3x3. The board starts as all 0's and a index contains a 1 if we placed a block there. A row if filled if it contains all 1's. The agent must learn to correctly place the blocks in the right space in order fill a row. A random piece is drawn from a set of possible pieces of size 1x1, 1x2, 1x3, and 2x1.

The agent does not have the option to rotate and there is no time or dropdown. The sign appears and the agent must find the best action to maximise reward.

### Observation space
The oberservation space is all the possible board layouts with all possible pieces. A single observation is a board layout and the current piece. Each cell is represented as an int derived from its row index * 3 + column index. Each board state is represented as the summation of 2 ** cell_id for each filled cell with each possible board state receiving a separate index. An observation is derived from a board state's index * 4 + the index of a piece.

### Action Space
The action space is the number of colums that the agent can choose from. In our case `cols=3` so the action space is 3. 

### Reward structure 
-1 for just placing a piece,
+10 per row clear * (number of rows cleared)**2, ie each row that only contains `1`'s,
-20 for game over

### Starting state, termination, and truncation conditions
The starting state is the board empty (all zeros) and some initial random piece.
Termination happens when a piece cannot be placed at the top of the board.
Truncation happens after 300 steps.

### Transition Noise
We have 4 possible pieces that the agent can see. These are random. 

### Arguments to __init__ and the registered environment id
rows: int = 3, cols: int = 3. This defines the grid size. render_mode, which mode we are rendering in.
