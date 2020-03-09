
import torch
import torch.nn.functional as F

from rlpyt.models.conv2d import Conv2dHeadModel

from intrinsic_rl.models.submodules import Identity, Conv2dHeadModelFlex



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
            conv_nonlinearity=Identity,
            mlp_nonlinearity=Identity,
            use_maxpool=False,
            base_model=None
            ):
        """Instantiate basic feature extractor. Uses base_model convolutional front-end, if available."""
        super().__init__()
        if base_model:
            for attr_key in dir(base_model):  # Looks for conv head model to share, assumes only one exists
                attr = getattr(base_model, attr_key)
                attr_cls = type(attr)
                if issubclass(attr_cls, Conv2dHeadModel) or issubclass(attr_cls, Conv2dHeadModelFlex):
                    self.extractor = attr
                    return
            raise AttributeError("Policy does not contain a convolutional head model to share")
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
