"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, rows: int, cols: int, piece_width: int, render_mode: str | None = None
    ):
        # num of rows and cols
        self.rows = rows
        self.cols = cols
        # piece width
        self.piece_width = piece_width

        # our board
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        # current piece the agent sees
        self.current_piece = 0

        # possible pieces for our agent to see
        self.pieces = [
            np.array([[1, 0], [1, 1]], dtype=np.int32),  # L
            np.array([[1, 0], [1, 0]], dtype=np.int32),  # I
            np.array([[1, 1], [0, 1]], dtype=np.int32),  # upside Down L
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiBinary((self.rows, self.cols)),
                "piece": gym.spaces.Discrete(3),
            }
        )
        # which column we choose the agent to drop the piece in
        # all availabel colums except the ones that would cause out of bounds on the right side
        # we will assume that the far left piece is the col we choose.
        self.action_space = gym.spaces.Discrete(self.cols - self.piece_width + 1)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        # pygame stuff (https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"piece": self.current_piece, "board": self.board.copy()}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {}

    def _get_reward(self):
        """
        Caculate the reward based on the number of rows cleared. if no rows cleared
        then we get negative reward
        """
        cleared = []
        # get index of rows that have been cleared
        for i in range(len(self.board)):
            row = self.board[i]
            if sum(row) == self.cols:
                cleared.append(i)

        # clear them
        for i in range(len(cleared)):
            self.board = np.delete(self.board, cleared[i], axis=0)
            new_rows = np.zeros((1, self.cols), dtype=np.int32)
            self.board = np.append(new_rows, self.board, axis=0)

        return len(cleared) if len(cleared) > 0 else -10

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # set the board to empty and set the piece to random piece
        self.board.fill(0)
        self.current_piece = int(self.np_random.integers(3))
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def can_place(self, piece, top_row, col):
        """
        Check if we can place a piece
        """
        for piece_row in range(piece.shape[0]):
            for piece_col in range(piece.shape[1]):
                if piece[piece_row, piece_col] == 1:
                    board_row = top_row + piece_row
                    board_col = col + piece_col

                    if board_row >= self.rows:
                        return False
                    if self.board[board_row, board_col] == 1:
                        return False
        return True

    def place(self, piece, col: int):
        """
        Place the piece on the board
        """
        top_row = 0
        # if we cannot place return
        if not self.can_place(piece, top_row, col):
            return False

        # check row by row where we can place
        while self.can_place(piece, top_row + 1, col):
            top_row += 1

        # place the row on board itself
        for piece_row in range(piece.shape[0]):
            for piece_col in range(piece.shape[1]):
                if piece[piece_row, piece_col] == 1:
                    self.board[top_row + piece_row, col + piece_col] = 1

        return True

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.
        # get the piece
        piece = self.pieces[self.current_piece]
        # place it
        terminated = not self.place(piece, action)
        # get the reward
        reward = self._get_reward()

        if self.render_mode == "human":
            self._render_frame()
        # set a new current piece
        self.current_piece = int(self.np_random.integers(len(self.pieces)))
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        raise NotImplementedError

    def _render_frame(self):
        """
        pygame stuff for rendering. Used (https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
        as reference/template and updated to fit our env.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.cols
        )  # The size of a single grid square in pixels

        # draw fillde cells on board
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 120, 255),
                        pygame.Rect(
                            col * pix_square_size,
                            row * pix_square_size,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )

        # draw a rows on a grid
        for row in range(self.rows + 1):
            y = row * pix_square_size
            pygame.draw.line(
                canvas, (0, 0, 0), (0, y), (self.cols * pix_square_size, y), width=2
            )
        # draw cols on a grid
        for col in range(self.cols + 1):
            x = col * pix_square_size
            pygame.draw.line(
                canvas, (0, 0, 0), (x, 0), (x, self.rows * pix_square_size), width=2
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/MyEnv-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
from gymnasium.utils.env_checker import check_env

if __name__ == "__main__":
    # This will catch many common issues
    env = MyEnv(rows=4, cols=4, piece_width=2, render_mode="human")
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    print(env)
    env.reset()
    obs, info = env.reset()
    print(f"OBSERVATION: {obs}")
    print(f"PICE: {env.pieces[env.current_piece]}")
    print(f"INFO: {info}")
    running = True

    # random test loop
    for i in range(100):
        action = int(env.np_random.integers(len(env.pieces)))
        obs, reward, terminated, truncated, info = env.step(action=action)
        print(f"index: {i}")
        print(f"ACTION: {action}")
        print(f"PICE: {env.pieces[env.current_piece]}")
        print(f"REWARD: {reward}")
        print(f"TERMINATED: {terminated}")
        print(f"TRUNACTED: {truncated}")
        print(f"INFO: {info}")
        print(f"OBSERVATION: {obs}")
        print()

        if terminated:
            break

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        env._render_frame()
    env.close()
