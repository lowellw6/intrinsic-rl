
import numpy as np
from collections import namedtuple

from rlpyt.envs.atari.atari_env import AtariTrajInfo, AtariEnv
from rlpyt.envs.base import EnvStep

MontezumaEnvInfo = namedtuple("MontezumaEnvInfo", ["game_score", "traj_done", "room_id"])


class MontezumaTrajInfo(AtariTrajInfo):
    """Adds reporting of rooms visited."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.RoomsVisited = set()

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.RoomsVisited.add(env_info.room_id)

    def terminate(self, observation):
        """Locks in rooms visited count as int (rather than set) for logger stats."""
        self.RoomsVisited = len(self.RoomsVisited)
        return self


class MontezumaEnv(AtariEnv):
    """Adds tracking of rooms visited."""

    def step(self, action):
        a = self._action_set[action]
        game_score = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)
        self._get_screen(1)
        game_score += self.ale.act(a)
        lost_life = self._check_life()  # Advances from lost_life state.
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
        self._update_obs()
        reward = np.sign(game_score) if self._clip_reward else game_score
        game_over = self.ale.game_over() or self._step_counter >= self.horizon
        done = game_over or (self._episodic_lives and lost_life)
        # Include reporting of current room ID in Montezuma Revenge (stored at RAM address 3)
        info = MontezumaEnvInfo(game_score=game_score, traj_done=game_over, room_id=self.ale.getRAM()[3])
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, done, info)