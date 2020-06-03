
import torch
import torch.nn.init as init

from rlpyt.models.running_mean_std import RunningMeanStdModel

from intrinsic_rl.models.bonus.base import SelfSupervisedModule
from intrinsic_rl.models.bonus.feat_embed import BaseFeatureExtractor
from intrinsic_rl.util import wrap


def rnd_param_init_(module):
    """
    Initializes RND module weights and biases to match that in:
    https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py.

    The authors initialize all conv and dense weights orthogonally with gain sqrt(2),
    and constant initialize all conv and dense biases to 0.
    """
    for name, parameter in module.named_parameters():
        tokens = name.split('.')
        if "weight" in tokens:
            init.orthogonal_(parameter, gain=(2 ** 0.5))
        elif "bias" in tokens:
            init.constant_(parameter, val=0.)


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
        rnd_param_init_(self.target_model)
        rnd_param_init_(self.distill_model)
        self.obs_rms = RunningMeanStdModel(wrap(rnd_model_kwargs["input_shape"]))  # Requires RndCls takes input_shape
        self.int_rff = None  # Intrinsic reward forward filter (this stores a discounted sum of non-episodic rewards)
        self.int_rff_rms = RunningMeanStdModel(torch.Size([1]))  # Intrinsic reward forward filter RMS model
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
        obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-5)
        obs = torch.clamp(obs, min=-5, max=5)
        return obs

    def normalize_int_rew(self, int_rew, gamma=0.99):
        """
        Normalizes intrinsic rewards according to specifications in
        https://arxiv.org/abs/1810.12894. This is done to remove the
        need to search for optimal intrinsic reward scaling factors in between
        different environments.

        This model is *not* expected to be initialized, if following the authors'
        implementation.
        """
        # Update rewards forward filter and gather batch of results
        rff_batch = torch.empty_like(int_rew)
        int_rff_prior = self.int_rff
        for i, rews in enumerate(int_rew):
            if self.int_rff is None:
                self.int_rff = rews
            else:
                self.int_rff = self.int_rff * gamma + rews
            rff_batch[i, :] = self.int_rff

        # Update intrinsic rff rms for int rew normalization if updating norm models
        if self.update_norm:
            batch_size = rff_batch.numel()
            self.int_rff_rms.update(rff_batch.view((batch_size, 1)))
        else:  # Reset rff prior state if not updating norm models
            self.int_rff = int_rff_prior

        # Normalize by dividing out running std of rff values
        return int_rew / torch.sqrt(self.int_rff_rms.var)

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
        int_rew = pred_errors.detach()
        return int_rew, distill_loss
