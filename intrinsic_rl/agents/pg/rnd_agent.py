
from rlpyt.agents.pg.atari import AtariFfAgent

from intrinsic_rl.agents.pg.base import IntrinsicPolicyGradientAgent
from intrinsic_rl.models.pg.atari_models import IntValAtariFfModel
from intrinsic_rl.models.bonus.rnd import RndBonusModule
from intrinsic_rl.models.bonus.feat_embed import ConvFeatureExtractor


class RndAtariFfAgent(IntrinsicPolicyGradientAgent, AtariFfAgent):
    """
    Atari agent for non-recurrent policy gradient algorithms
    which utilize a Random Network Distillation intrinsic bonus.

    A ConvFeatureExtractor is used for both the target and distilled
    networks. The user should specify all inputs to this model in
    ``rnd_model_kwargs`` except for ``input_shape``, which is
    automatically determined from the environment.
    """

    def __init__(self, rnd_model_kwargs, ModelCls=IntValAtariFfModel, **kwargs):
        if "input_shape" in rnd_model_kwargs:
            raise AttributeError("Param ``input_shape`` automatically determined from environment; "
                                 "specify all RND network params except ``input_shape``.")
        bonus_model_kwargs = dict(RndCls=ConvFeatureExtractor, rnd_model_kwargs=rnd_model_kwargs)
        super().__init__(BonusModelCls=RndBonusModule, bonus_model_kwargs=bonus_model_kwargs,
                         ModelCls=ModelCls, **kwargs)

    def extract_bonus_inputs(self, next_observation, **kwargs):
        """Extracts and returns batched last frames within frame-stack."""
        next_observation = next_observation[:, -1].unsqueeze(dim=1)
        return next_observation

    def add_env_to_bonus_kwargs(self):
        """
        Adds input_shape and output_size to rnd_model_kwargs, taken from environment info.
        Frame-stack dimension becomes singular as RND model only uses last frame.
        """
        rnd_input_shape = tuple([1] + list(self.env_model_kwargs["image_shape"][1:]))  # (1, H, W)
        self.bonus_model_kwargs["rnd_model_kwargs"]["input_shape"] = rnd_input_shape
