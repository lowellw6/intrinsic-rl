
import torch
from abc import ABC, abstractmethod

from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel

from intrinsic_rl.models.base import SelfSupervisedModule
from intrinsic_rl.models.submodules import Conv2dHeadModelFlex
from intrinsic_rl.util import trimSeq


class BaseFeatureExtractor(SelfSupervisedModule, ABC):
    """
    Abstract base feature extraction class for mapping observations
    into feature embeddings. Inheriting classes must define the module
    ``self.extractor`` to produce this mapping.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input):
        """
        Maps input observations into a lower dimensional feature space.
        This mapping function can be constant or updating alongside the baseline model.
        Assumes leading batch dimension.

        Takes an arbitrary number of observation tensors. So long as the non-batch dimensions
        match (those after the first), these tensors are concatenated into one large batch for the
        forward pass and are split into components again afterward.
        """
        batch_lens = [obs.shape[0] for obs in input]  # Will probably be same size for each
        obs_cat = torch.cat([obs for obs in input], dim=0)
        obs_feat_pile = self.extractor(obs_cat)
        obs_feat_splits = torch.split(obs_feat_pile, batch_lens, dim=0)
        return trimSeq(obs_feat_splits), torch.zeros(1)  # Zero loss, no inherent gradient update


class IdentityFeatureExtractor(BaseFeatureExtractor):
    """Wraps torch.nn.Identity to follow self-supervised module loss provision protocol."""

    def __init__(self):
        super().__init__()
        self.extractor = torch.nn.Identity()


class ConvFeatureExtractor(BaseFeatureExtractor):
    """
    Maps images to feature embedding with convolutional front end
    which is randomly initialized.
    """

    def __init__(
            self,
            image_shape=None,
            channels=None,
            kernel_sizes=None,
            strides=None,
            hidden_sizes=None,
            output_size=None,
            paddings=None,
            conv_nonlinearity=torch.nn.Identity,
            mlp_nonlinearity=torch.nn.Identity,
            use_maxpool=False,
            decision_model=None  # Base policy or q-value function
            ):
        """Instantiate conv feature extractor. Uses decision_model convolutional front-end, if available."""
        super().__init__()
        if decision_model:
            for attr_key in dir(decision_model):  # Looks for conv head model to share, assumes only one exists
                attr = getattr(decision_model, attr_key)
                attr_cls = type(attr)
                if issubclass(attr_cls, Conv2dHeadModel) or issubclass(attr_cls, Conv2dHeadModelFlex):
                    self.extractor = attr
                    return
            raise AttributeError("Base policy / q-network does not contain a convolutional head model to share")
        else:
            assert None not in (image_shape, channels, kernel_sizes, strides, hidden_sizes)
            self.extractor = Conv2dHeadModelFlex(
                image_shape=image_shape,
                channels=channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                paddings=paddings,
                conv_nonlinearity=conv_nonlinearity,
                mlp_nonlinearity=mlp_nonlinearity,
                use_maxpool=use_maxpool
            )


class MlpFeatureExtractor(BaseFeatureExtractor):
    """
    Maps 1D state vectors to feature embedding with MLP.
    Functionally equivalent to ConvFeatureExtractor but for non-image states
    where feature extraction is still necessary (e.g. use with Inverse Dynamics
    Feature Extraction or with Random Network Distillation).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size=None,
            nonlinearity=torch.nn.Identity
            ):
        """Instantiate MLP feature extractor. Does not support parameter sharing with base network."""
        super().__init__()
        self.extractor = MlpModel(input_size, hidden_sizes, output_size, nonlinearity)


class InverseDynamicsFeatureExtractor(SelfSupervisedModule):  # TODO
    pass