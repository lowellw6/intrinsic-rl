
from intrinsic_rl.algos.pg.intrinsic_ppo import IntrinsicPPO


class RndIntrinsicPPO(IntrinsicPPO):
    next_obs = True  # Tells sampler it needs next observations
