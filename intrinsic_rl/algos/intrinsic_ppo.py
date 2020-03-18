
import torch
from abc import ABC, abstractmethod

from rlpyt.algos.utils import discount_return, generalized_advantage_estimation
from rlpyt.algos.pg.base import OptInfo
from rlpyt.algos.pg.ppo import PPO, LossInputs
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.misc import iterate_mb_idxs


class IntrinsicPPO(PPO, ABC):
    """
    ###
    """

    def __init__(self,
                 int_discount=0.99,  # Separate discount factor for intrinsic reward stream
                 int_rew_coeff=1.,
                 ext_rew_coeff=0.,
                 bonus_loss_coeff=1.,
                 entropy_loss_coeff=0.,  # Default is to discard policy entropy
                 **kwargs):
        save__init__args(locals())
        super().__init__(entropy_loss_coeff=entropy_loss_coeff, **kwargs)

    @abstractmethod
    def extract_bonus_inputs(self, samples, **other_input_kwargs):
        """Should extract and format all necessary inputs to the bonus model."""
        pass

    @abstractmethod
    def preprocess_obs(self, samples, **other_obs_kwargs):
        """Run any algorithm specific observation preprocessing, such as normalization."""
        pass

    def process_intrinsic_returns(self, int_rew, value, bootstrap_value):
        """
        Same as ``process_returns`` but discounted reward signal is carried over episodes.
        Note that value and bootstrap_value should come from separate critic model than that
        used for extrinsic rewards to keep these reward streams distinct.
        For more details, see https://arxiv.org/abs/1810.12894.
        """
        faux_done = torch.zeros_like(int_rew)  # Faux done signals, all "not done"

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(int_rew, faux_done, bootstrap_value, self.int_discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                int_rew, value, faux_done, bootstrap_value, self.int_discount, self.gae_lambda)

        if self.normalize_advantage:
            adv_mean = advantage.mean()
            adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage

    def optimize_agent(self, itr, samples):
        """
        ###
        """

