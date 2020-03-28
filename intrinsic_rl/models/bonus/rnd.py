
import torch

from rlpyt.models.running_mean_std import RunningMeanStdModel

from intrinsic_rl.models.bonus.base import SelfSupervisedModule
from intrinsic_rl.models.bonus.feat_embed import BaseFeatureExtractor


class RndBonusModule(SelfSupervisedModule):
    """
    Random Network Distillation Module. Produces intrinsic
    rewards as the prediction error between the feature
    embeddings from a target and distilled model, both
    randomly initialized.
    """

    def __init__(
            self,
            RndCls,  # type: BaseFeatureExtractor
            rnd_model_kwargs
            ):
        """
        Constructs target and distillation model. Assumes identical architectures.
        Also constructs observation normalization model.
        """
        super().__init__()
        self.target_model = RndCls(**rnd_model_kwargs)
        self.distill_model = RndCls(**rnd_model_kwargs)
        self.obs_norm_model = RunningMeanStdModel(rnd_model_kwargs["input_shape"])  # Requires RndCls takes input_shape

    def normalize_obs(self, obs):
        """
        Normalizes observations according to specifications in
        https://arxiv.org/abs/1810.12894. This is necessary since the target
        network is fixed and cannot adjust to varying environments.

        WARNING: If observations are already normalized using
        a different model / formulation, this can cause issues
        if this model is initialized in the sampler.
        """
        self.obs_norm_model.update(obs.to(dtype=torch.float32))
        obs = (obs - self.obs_norm_model.mean) / (self.obs_norm_model.var.sqrt() + 1e-8)
        obs = torch.clamp(obs, min=-5, max=5)
        return obs

    def forward(self, obs):
        """
        Runs forward pass for distillation and target models, producing intrinsic
        bonuses and distillation model loss. Note the self-supervised losses of
        the models are unused (and are presumably placeholders with a value of zero).
        """
        obs = self.normalize_obs(obs)
        distill_feat, _ = self.distill_model(obs)
        target_feat, _ = self.target_model(obs)
        pred_errors = torch.mean((distill_feat - target_feat.detach()) ** 2, dim=-1)  # Maintains batch dimension
        distill_loss = torch.mean(pred_errors)  # Reduces batch dimension
        int_rew = pred_errors.detach()
        return int_rew, distill_loss
