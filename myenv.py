"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register
from numpy.core.numeric import int32


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, rows: int, cols: int, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.
        self.rows = rows
        self.cols = cols

        # init positions. reset will handle the actual positions
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_locations = np.array(
            [[-1, -1] for i in range(cols)], dtype=np.int32
        )

        self.observation_space = ...  # TODO: implement this
        # two possible actions, move left or right
        self.action_space = gym.spaces.Discrete(2)

        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right (column + 1)
            2: np.array([0, -1]),  # Move left (column - 1)
        }

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
        return {"agent": self._agent_location, "target": self._target_locations}

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

        # set agent state top in middle col at top
        self._agent_location = np.array([0, self.cols // 2], dtype=np.int32)
        # target locations are the bottom row
        self._target_locations = np.array(
            [[-1, i] for i in range(self.cols)], dtype=int32
        )
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

        terminated = False  # TODO: make so that if
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return
        # TODO: draw it. You need this for the sample episode in your report.
        raise NotImplementedError

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
