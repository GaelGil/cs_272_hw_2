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

    def __init__(self, rows: int, cols: int, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.
        self.rows = rows
        self.cols = cols

        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.current_piece = 0

        # possible pieces for our agent to see
        self.pieces = [
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int32),  # L
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int32),  # I
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int32),  # T
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.MultiBinary((self.rows, self.cols)),
                "piece": gym.spaces.Discrete(3),
            }
        )
        self.action_space = gym.spaces.Discrete(self.cols)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        # pygame stuff (https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
        self.window_size = 512  # The size of the PyGame window
        self.window = None

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
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_locations, ord=1
            )
        }

    def _get_reward(self):
        pass

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        self.board.fill(0)
        self.current_piece = int(self.np_random.integers(3))
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.
        direction = self._action_to_direction[action]

        # update agent location
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.cols - 1
        )

        terminated = False  # TODO: make so that if we get to top of grid its over
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()  # TODO: implement this, see notes.txt
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        raise NotImplementedError

    def _render_frame(self):
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

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_locations[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.cols + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
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

# This will catch many common issues
env = MyEnv(rows=5, cols=6)
print(env)
env.reset()
print(env._get_obs())
print(env._get_obs()["agent"].shape, env._get_obs()["agent"].dtype)
print(env._get_obs()["target"].shape, env._get_obs()["target"].dtype)
print(env.observation_space.contains(env._get_obs))
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
