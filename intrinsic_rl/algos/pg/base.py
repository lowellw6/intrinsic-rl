
import torch
from abc import ABC

from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation

from intrinsic_rl.algos.base import IntrinsicRlAlgorithm


class IntrinsicPolicyGradientAlgo(PolicyGradientAlgo, IntrinsicRlAlgorithm, ABC):
    """
    Abstract class extending PolicyGradientAlgo to support interface
    with intrinsic bonus model.
    """

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
