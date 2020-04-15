
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
        Also constructs normalization models for observation and intrinsic rewards.
        """
        super().__init__()
        self.target_model = RndCls(**rnd_model_kwargs)
        self.distill_model = RndCls(**rnd_model_kwargs)
        self.obs_rms = RunningMeanStdModel(rnd_model_kwargs["input_shape"])  # Requires RndCls takes input_shape
        self.int_rew_rms = RunningMeanStdModel(torch.Size([1]))
        self.update_norm = True  # Default to updating obs and int_rew normalization models

    def normalize_obs(self, obs):
        """
        Normalizes observations according to specifications in
        https://arxiv.org/abs/1810.12894. This is necessary since the target
        network is fixed and cannot adjust to varying environments.

        This model should be initialized in the sampler by running
        a small number of observations through it.

        WARNING: If observations are already normalized using
        a different model / formulation, this will cause issues
        if this model is initialized on raw obs in the sampler.
        """
        obs = obs.to(dtype=torch.float32)  # Obs may be byte tensor (e.g. 8-bit pixels)
        if self.update_norm:
            self.obs_rms.update(obs)
        obs = (obs - self.obs_rms.mean) / (self.obs_rms.var.sqrt() + 1e-8)
        obs = torch.clamp(obs, min=-5, max=5)
        return obs

    def normalize_int_rew(self, int_rew):
        """
        Normalizes intrinsic rewards according to specifications in
        https://arxiv.org/abs/1810.12894. This is done to remove the
        need to search for optimal intrinsic reward scaling factors in between
        different environments.

        This model is *not* expected to be initialized, if following the authors'
        implementation.
        """
        batch_size = int_rew.numel()
        int_rew = int_rew.view((batch_size, 1))  # Ensures a non-batch dimension for RunningMeanStdModel
        if self.update_norm:
            self.int_rew_rms.update(int_rew)
        int_rew /= self.int_rew_rms.var.sqrt()
        return int_rew.squeeze()

    def forward(self, next_obs):
        """
        Runs forward pass for distillation and target models, producing intrinsic
        bonuses and distillation model loss. Note the self-supervised losses of
        the models are unused (and are presumably placeholders with a value of zero).
        """
        next_obs = self.normalize_obs(next_obs)
        distill_feat, _ = self.distill_model(next_obs)
        target_feat, _ = self.target_model(next_obs)
        pred_errors = torch.mean((distill_feat - target_feat.detach()) ** 2, dim=-1)  # Maintains batch dimension
        distill_loss = torch.mean(pred_errors)  # Reduces batch dimension
        int_rew = self.normalize_int_rew(pred_errors.detach())
        return int_rew, distill_loss, next_obs  # Returns normalized next obs for logging
