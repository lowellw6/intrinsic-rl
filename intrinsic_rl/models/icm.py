
from intrinsic_rl.models.base import SelfSupervisedModule
from intrinsic_rl.models.feat_embed import BaseFeatureExtractor, IdentityFeatureExtractor


class IntrinsicCuriosityModule(SelfSupervisedModule):
    """
    Fully self-supervised intrinsic curiosity module.
    Produces intrinsic rewards for a given batch of transitions
    using the loss from a forward dynamics module. Note this module
    can contain more than one individual forward dynamics models (FDMs),
    as is the case when using ForwardDisagreementEnsemble.

    This module may or may not use a feature extractor to map high
    dimensional observations to an intermediate feature embedding, which
    constitutes the FDM's observation input space if so.
    """

    def __init__(
            self,
            ForwardDynamicsCls,  # type: SelfSupervisedModule
            forward_model_kwargs,
            forward_loss_coeff=1.,
            FeatExtractCls=None,  # type: BaseFeatureExtractor
            feat_extract_kwargs=None,
            feat_loss_coeff=1.
            ):
        super().__init__()
        self.fwd_dyn_module = ForwardDynamicsCls(**forward_model_kwargs)
        self.forward_loss_coeff = forward_loss_coeff
        if not FeatExtractCls:  # Default to no-op feature extraction
            assert not feat_extract_kwargs
            self.feat_extractor = IdentityFeatureExtractor()
        else:
            assert isinstance(feat_extract_kwargs, dict)
            self.feat_extractor = FeatExtractCls(**feat_extract_kwargs)
        self.feat_loss_coeff = feat_loss_coeff

    def forward(self, obs, action, next_obs):
        """
        Produces an intrinsic reward and the loss for this bonus model.
        The intrinsic reward is the loss of the forward dynamics module.
        """
        features, feat_extract_loss = self.feat_extractor(obs, next_obs)
        obs_feat, next_obs_feat = features
        fwd_out, fwd_dyn_loss = self.fwd_dyn_module(obs_feat.detach(), action.detach(), next_obs_feat.detach())
        icm_loss = (self.feat_loss_coeff * feat_extract_loss) + (self.forward_loss_coeff * fwd_dyn_loss)
        # Don't care about prediction itself from FDM class; just take the bonuses and loss
        _, fwd_bonus = fwd_out
        int_rew = fwd_bonus.detach()
        return int_rew, icm_loss
