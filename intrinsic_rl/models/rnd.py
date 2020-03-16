
import torch

from intrinsic_rl.models.base import SelfSupervisedModule
from intrinsic_rl.models.feat_embed import BaseFeatureExtractor


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
        """Constructs target and distillation model. Assumes identical architectures."""
        super().__init__()
        self.target_model = RndCls(**rnd_model_kwargs)
        self.distill_model = RndCls(**rnd_model_kwargs)

    def forward(self, obs):
        """
        Runs forward pass for distillation and target models, producing intrinsic
        bonuses and distillation model loss. Note the self-supervised losses of
        the models are unused (and are presumably placeholders with a value of zero).
        """
        distill_feat, _ = self.distill_model(obs)
        target_feat, _ = self.target_model(obs)
        pred_errors = torch.mean((distill_feat - target_feat.detach()) ** 2, dim=-1)  # Maintains batch dimension
        distill_loss = torch.mean(pred_errors)  # Reduces batch dimension
        int_rew = pred_errors.detach()
        return int_rew, distill_loss
