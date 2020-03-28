
import torch
from abc import ABC

from rlpyt.algos.pg.base import PolicyGradientAlgo
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation, valid_from_done

from intrinsic_rl.algos.base import IntrinsicRlAlgorithm


class IntrinsicPolicyGradientAlgo(PolicyGradientAlgo, IntrinsicRlAlgorithm, ABC):
    """
    Abstract class extending PolicyGradientAlgo to support interface
    with intrinsic bonus model.
    """

    def process_returns(self, samples):
        raise Exception("Method split into extrinsic and intrinsic components, "
                        "see IntrinsicPolicyGradientAlgo class definition.")

    def process_extrinsic_returns(self, samples):
        """
        Identical to ``process_returns`` but calls renamed ``ext_value`` in
        agent buffer (was named ``value``). This was renamed since there
        is now an ``int_value`` corresponding to the intrinsic value head.
        """
        reward, done, value, bv = (samples.env.reward, samples.env.done,
            samples.agent.agent_info.ext_value, samples.agent.bootstrap_value)  # Rename edit here
        done = done.type(reward.dtype)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage, valid

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
