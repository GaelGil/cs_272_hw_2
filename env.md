The story: what the world is and what the agent is trying to do
Observation space, including exactly how you encode the state into one integer
Action space
Reward structure
Starting state, termination, and truncation conditions
The transition noise: what is random, and with what probability
Arguments to __init__ and the registered environment id

### The Story
The world is a simplified version of tetris. The agent is trying to find the best columns to place blocks in order to get a row equal to its length. ie if we have rows length 3. The agent must learn to correctly place the blocks in the right space in order. 

The agent does not have the option to rotate and their is no time or dropdown. The sign appears and the agent must find the best action to maximise reward. 

### Observation space

### Action Space
The action space is the number of colums that the agent can choose. In our case `cols=3` so the action space is 3. 


### Reward structure 
-1 for just placing a piece,
10 * 2 for each row cleared


### Starting state, termination, and truncation conditions
The starting state is the board empty and some initial random piece.
Termination happens when a piece cannot be placed at the top of the board. 

### Transition Noise
We have 4 possible pieces that the agent can see. These are random. 

### Arguments to __init__ and the registered environment id
rows: int = 3, cols: int = 3. This defines the grid size. 

 
 render_mode: str | None = None
