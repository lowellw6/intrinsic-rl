
import torch
import torch.nn.functional as F

from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class IntValAtariFfModel(AtariFfModel):
    """
    Adds a separate value head to AtariFfModel to be used for
    an intrinsic reward stream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.int_value = torch.nn.Linear(self.conv.output_size, 1)

    def forward(self, image, prev_action, prev_reward):
        """
        Overrides AtariFfModel forward to also run separate
        intrinsic value head.
        """
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        ext_val = self.value(fc_out).squeeze(-1)
        int_val = self.int_value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, ext_val, int_val = restore_leading_dims((pi, ext_val, int_val), lead_dim, T, B)

        return pi, ext_val, int_val
