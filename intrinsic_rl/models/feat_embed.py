
import torch

from rlpyt.models.conv2d import Conv2dHeadModel

from intrinsic_rl.models.submodules import Conv2dHeadModelFlex


class BasicFeatureExtractor(torch.nn.Module):
    """
    Maps images to feature embedding with constant convolutional front end
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
        """Instantiate basic feature extractor. Uses decision_model convolutional front-end, if available."""
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

    def forward(self, input):
        """
        Maps input observations into a lower dimensional feature space.
        This mapping function can be constant or updating alongside the baseline model.
        Assumes input shape: [B,C,H,W]. (i.e. images)
        """
        return self.extractor(input)
