
from rlpyt.agents.pg.atari import AtariFfAgent

from intrinsic_rl.agents.base import IntrinsicBonusAgent
from intrinsic_rl.models.rnd import RndBonusModule
from intrinsic_rl.models.feat_embed import ConvFeatureExtractor


class RndAtariFfAgent(IntrinsicBonusAgent, AtariFfAgent):
    """
    Atari agent for non-recurrent policy gradient algorithms
    which utilize a Random Network Distillation intrinsic bonus.

    A ConvFeatureExtractor is used for both the target and distilled
    networks. The user should specify all inputs to this model in
    ``rnd_model_kwargs`` except for ``image_shape``, which is
    automatically determined from the environment.
    """

    def __init__(self, rnd_model_kwargs, **kwargs):
        if "image_shape" in rnd_model_kwargs:
            raise AttributeError("Param ``image_shape`` automatically determined from environment; "
                                 "specify all RND network params except ``image_shape``.")
        bonus_model_kwargs = dict(RndCls=ConvFeatureExtractor, rnd_model_kwargs=rnd_model_kwargs)
        super().__init__(BonusModelCls=RndBonusModule, bonus_model_kwargs=bonus_model_kwargs, **kwargs)

    def add_env_to_bonus_kwargs(self):
        """Adds image_shape and output_size to rnd_model_kwargs, taken from environment info."""
        self.bonus_model_kwargs["rnd_model_kwargs"]["image_shape"] = self.env_model_kwargs["image_shape"]
