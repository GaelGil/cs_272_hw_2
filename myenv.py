"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, rows: int = 3, cols: int = 3, render_mode: str | None = None):
        # num of rows and cols
        self.rows = rows
        self.cols = cols

        # our board
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        # current piece the agent sees
        self.current_piece = 0

        # possible pieces for our agent to see
        self.pieces = [
            np.array([[1]], dtype=np.int32),  # 1x1
            np.array([[1], [1]], dtype=np.int32),  # 1x2
            np.array([[1], [1], [1]], dtype=np.int32),  # 1x3
            np.array([[1, 1]], dtype=np.int32),  # 2x1
        ]

        # columns agent can choose
        self.action_space = gym.spaces.Discrete(self.cols)

        # all possible board layouts
        self.boards = self._enum_boards()

        self.board_index = {board_id: i for i, board_id in enumerate(self.boards)}

        # number of board layouts * num pieces
        self.observation_space = gym.spaces.Discrete(
            len(self.boards) * len(self.pieces)
        )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    def _board_to_int(self, board):
        """Convert board state to int: row * cols + col"""
        board_id = 0
        cell_index = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == 1:
                    board_id += 2**cell_index
                cell_index += 1
        return board_id

    def _int_to_board(self, board_id):
        """Convert board_id back into board state obj"""
        board = np.zeros((self.rows, self.cols), dtype=np.int32)
        for row in range(self.rows):
            for col in range(self.cols):
                board[row][col] = board_id % 2
                board_id = board_id // 2
        return board

    def _enum_boards(self):
        """BFS to enumerate all possible boardstates.

        Returns:
            list[int]: Sorted ints representing each board state
        """

        temp_board = self.board
        queue = deque([0])
        boards_seen = {0}

        while queue:
            current_board_id = queue.popleft()
            for piece in self.pieces:
                for col in range(self.action_space.n):
                    self.board = self._int_to_board(current_board_id)
                    if self.place(piece, col):
                        self._get_reward()
                        new_board_id = self._board_to_int(self.board)
                        if new_board_id not in boards_seen:
                            boards_seen.add(new_board_id)
                            queue.append(new_board_id)
        self.board = temp_board
        return sorted(boards_seen)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            int: board_index * pieces + current piece
        """
        board_id = self._board_to_int(self.board)
        return self.board_index[board_id] * len(self.pieces) + self.current_piece

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {"board": self.board.copy(), "piece": self.current_piece}

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

        # delete rows from the bottom
        for index in reversed(cleared):
            self.board = np.delete(self.board, index, axis=0)

        # add empty rows to top
        if cleared:
            empty_rows = np.zeros((len(cleared), self.cols), dtype=np.int32)
            self.board = np.vstack((empty_rows, self.board))

        return (
            -1 + 10 * len(cleared) ** 2
        )  # -1 place piece, (+10 line clear) ** 2 per line

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # set the board to empty and set the piece to random piece
        self.board.fill(0)
        self.current_piece = int(self.np_random.integers(len(self.pieces)))
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
        col = min(
            col, self.cols - piece.shape[1]
        )  # prevents 2x1 piece from being placed out of bounds
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

        if terminated:
            reward = -20
        else:
            # get the reward
            reward = self._get_reward()
            # set a new current piece
            self.current_piece = int(self.np_random.integers(len(self.pieces)))

        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Return a text representation when ANSI rendering is enabled."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self):
        """Render board as ansi"""
        ansi_board = f"#{'=' * (self.cols * 3)}#\n"

        for row in range(self.rows):
            ansi_board += "|"
            for col in range(self.cols):
                if self.board[row][col] == 1:
                    ansi_board += " 1 "
                else:
                    ansi_board += " 0 "
            ansi_board += "|\n"
        ansi_board += f"#{'=' * (self.cols * 3)}#\n"
        piece = self.pieces[self.current_piece]
        ansi_board += f"Current Piece: {piece.shape[1]}x{piece.shape[0]}\n"
        return ansi_board

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
    render_mode = "ansi"
    env = MyEnv(render_mode=render_mode)
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
    if env.render_mode == "ansi":
        print(env.render())
    # random test loop
    for i in range(100):
        action = int(env.np_random.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action=action)
        print(f"index: {i}")
        print(f"ACTION: {action}")
        print(f"PICE: {env.pieces[env.current_piece]}")
        print(f"REWARD: {reward}")
        print(f"TERMINATED: {terminated}")
        print(f"TRUNACTED: {truncated}")
        print(f"INFO: {info}")
        print(f"OBSERVATION: {obs}")
        if env.render_mode == "ansi":
            print(env.render())
        print()

        if terminated:
            break

    env.close()
