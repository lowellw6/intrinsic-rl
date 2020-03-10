
import torch

from intrinsic_rl.models.submodules import Conv2dHeadModelFlex, ResidualBlock


class ForwardDynamicsModel(torch.nn.Module):
    """
    Model for (state, action) --> (next state) mapping.
    This base model assumes high-dimensional states (e.g. images) are
    already mapped to a flattened embedded feature space.
    """

    def __init__(
            self,
            obs_feat_size,
            action_size,
            hidden_sizes=512,
            num_res_blocks=4,
            nonlinearity=torch.nn.Identity
            ):
        super().__init__()
        self.fc_in = torch.nn.Linear(obs_feat_size + action_size, hidden_sizes)
        self.activ_in = nonlinearity()
        self.res_blocks = [ResidualBlock(hidden_sizes, inject_size=action_size, nonlinearity=nonlinearity)
                           for _ in range(num_res_blocks)]
        self.fc_out = torch.nn.Linear(hidden_sizes + action_size, obs_feat_size)

    def forward(self, obs_feat, action):
        """
        Runs forward pass for prediction mapping (obs_feat, action) --> next_obs_feat.
        If the action space is discrete, ``action`` is assumed to already be in one-hot format.
        Action tensor is injected between each layer via concatenation to observation features
        following implementation from (Burda et al., Large-Scale Study of Curiosity-Driven Learning,
        https://arxiv.org/abs/1808.04355).
        """
        x = torch.cat((obs_feat, action), dim=-1)
        x = self.fc_in(x)
        x = self.activ_in(x)
        for res_block in self.res_blocks:
            x = res_block((x, action))
        x = torch.cat((x, action), dim=-1)
        x = self.fc_out(x)
        return x
