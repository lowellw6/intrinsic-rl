
from intrinsic_rl.algos.pg.intrinsic_ppo import IntrinsicPPO


class RndIntrinsicPPO(IntrinsicPPO):
    next_obs = True  # Tells sampler it needs next observations

    def process_intrinsic_returns(self, int_rew, int_val, int_bootstrap_value):
        """
        Overrides method in IntrinsicPolicyGradientAlgo to add update to
        intrinsic reward norm model at end of method. Note agent can turn
        this on or off using ``set_norm_update`` method.
        """
        int_return, int_adv = super().process_intrinsic_returns(int_rew, int_val, int_bootstrap_value)
        self.agent.bonus_model.update_int_ret_rms(int_return)
        return int_return, int_adv
