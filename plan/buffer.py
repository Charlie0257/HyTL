import numpy as np
import torch as th


class PlanLayerBuffer:
    def __init__(self):
        self._trajs = []
        self._ctrl_rewards = []
        self._log_probs = []

    def clean(self):
        self._trajs = []
        self._ctrl_rewards = []

    def add_traj(self, traj: np.ndarray):
        self._trajs.append(traj)

    def add_ctrl_reward(self, ctrl_reward: float):
        if len(self._trajs) == 0:
            # workaround for resuming between rollouts
            return
        self._ctrl_rewards.append(ctrl_reward)

    def add_log_prob(self, log_prob: th.Tensor):
        self._log_probs.append(log_prob)

    def is_empty(self) -> bool:
        return self._trajs == []

    def drop_incomplete(self):
        """
        Control reward will be collected at one task complete or timeout (in CtrlRewardCollector),
        but traj and log_prob will be collect at beginning (in Dispatcher)
        Call this function to avoid incomplete sample
        """
        self._trajs = self.trajs[:len(self._ctrl_rewards)]
        self._log_probs = self._log_probs[:len(self._ctrl_rewards)]

    @property
    def norm_ctrl_rewards(self):
        rews = self.ctrl_rewards
        rew_mean = rews.mean()
        rew_std = rews.std()

        return (rews - rew_mean) / (rew_std + 1e-5)

    @property
    def trajs(self) -> np.ndarray:
        return np.array(self._trajs)

    @property
    def ctrl_rewards(self) -> np.ndarray:
        return np.array(self._ctrl_rewards)

    @property
    def log_probs(self) -> th.Tensor:
        return th.cat(self._log_probs)
