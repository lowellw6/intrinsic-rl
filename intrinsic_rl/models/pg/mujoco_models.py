
import torch
import numpy as np

from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class IntValMujocoFfModel(MujocoFfModel):
    """
    Adds a separate value head to MujocoFfModel to be used for
    an intrinsic reward stream.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            **kwargs
            ):
        """Instantiate additional intrinsic value MLP."""
        super().__init__(observation_shape, action_size,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         **kwargs)
        input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or [64, 64]
        # Make intrinsic value MLP identical to extrinsic value MLP
        self.iv = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            nonlinearity=hidden_nonlinearity,
        )

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, and value estimates from input state. Includes
        value estimates of both distinct intrinsic and extrinsic returns. See
        rlpyt MujocoFfModel for more information on this function.
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        obs_flat = observation.view(T * B, -1)
        mu = self.mu(obs_flat)
        ev = self.v(obs_flat).squeeze(-1)
        iv = self.iv(obs_flat).squeeze(-1)  # Added intrinsic value MLP forward pass
        log_std = self.log_std.repeat(T * B, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, ev, iv = restore_leading_dims((mu, log_std, ev, iv), lead_dim, T, B)

        return mu, log_std, ev, iv
