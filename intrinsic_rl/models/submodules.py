
import torch

from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel


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
            conv_nonlinearity=torch.nn.Identity,
            mlp_nonlinearity=torch.nn.Identity,
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


class ResidualBlock(torch.nn.Module):
    """
    Fully connected two-layer residual block with configurable intermediate activation.
    Output size is fixed as input size since the result is added to the residual input.
    Allows for a tensor injection to take place between each layer, concatenating the
    tensor with the intermediate features.
    """

    def __init__(
            self,
            input_size,
            hidden_size=512,
            nonlinearity=torch.nn.Identity,
            inject_size=0  # Size of continual tensor injection for each stage, if any
            ):
        super().__init__()
        self.inject_size = inject_size
        self.fc1 = torch.nn.Linear(input_size + inject_size, hidden_size)
        self.activ = nonlinearity()
        self.fc2 = torch.nn.Linear(hidden_size + inject_size, input_size)

    def forward(self, input):
        """
        Runs specific forward pass depending on whether there is an injection tensor
        which is concatenated at each layer.
        """
        if self.inject_size > 0:
            assert len(input) == 2
            seed, inject = input
            return self.forward_injection(seed, inject)
        else:  # No injection
            assert isinstance(input, torch.Tensor)
            return self.forward_sequential(input)

    def forward_sequential(self, seed):
        """Sequential forward pass through layers."""
        x = self.fc1(seed)
        x = self.activ(x)
        x = self.fc2(x)
        return seed + x

    def forward_injection(self, seed, inject):
        """Forward pass where injection which is concatenated before each layer."""
        x = torch.cat((seed, inject), dim=-1)
        x = self.fc1(x)
        x = self.activ(x)
        x = torch.cat((x, inject), dim=-1)
        x = self.fc2(x)
        return seed + x
