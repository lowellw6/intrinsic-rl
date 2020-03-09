
import torch

from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel


class Identity(torch.nn.Sequential):
    """
    Stand-in for an identity no-op modules.
    """
    pass


class Conv2dHeadModelFlex(torch.nn.Module):
    """
    Replaces rlpyt's Conv2dHeadModel to provide greater flexibility.
    Allows for convolutional layers and MLP layers to have different activations.
    """

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            conv_nonlinearity=Identity,
            mlp_nonlinearity=Identity,
            use_maxpool=False,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=conv_nonlinearity,
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=mlp_nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size