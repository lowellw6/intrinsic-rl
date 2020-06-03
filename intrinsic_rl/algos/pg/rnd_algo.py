
from intrinsic_rl.algos.pg.intrinsic_ppo import IntrinsicPPO


class RndIntrinsicPPO(IntrinsicPPO):
    next_obs = True  # Tells sampler it needs next observations

    def process_intrinsic_returns(self, int_rew, int_val, int_bootstrap_value):
        """
        Overrides method in IntrinsicPolicyGradientAlgo to normalize intrinsic
        returns (which also updates rff and rff_rms models, see ``RndBonusModule``).
        Note agent can turn this on or off using ``set_norm_update`` method.
        """
        device_prior = int_rew.device
        int_rew = int_rew.to(self.agent.device)  # Move rews to agent device for int rew norm models
        int_rew = self.agent.bonus_model.normalize_int_rew(int_rew, gamma=self.int_discount)
        int_rew = int_rew.to(device_prior)  # Move normalized rews back to device in use outside call
        int_return, int_adv = super().process_intrinsic_returns(int_rew, int_val, int_bootstrap_value)
        return int_return, int_adv
