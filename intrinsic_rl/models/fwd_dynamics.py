
import torch

from intrinsic_rl.models.base import SelfSupervisedModule
from intrinsic_rl.models.submodules import ResidualBlock


class ForwardDynamicsModel(SelfSupervisedModule):
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
        res_block_list = [ResidualBlock(hidden_sizes, inject_size=action_size, nonlinearity=nonlinearity)
                          for _ in range(num_res_blocks)]
        self.res_blocks = torch.nn.ModuleList(res_block_list)
        self.fc_out = torch.nn.Linear(hidden_sizes + action_size, obs_feat_size)

    def forward(self, obs_feat, action, next_obs_feat):
        """
        Runs forward pass for prediction mapping (obs_feat, action) --> next_obs_feat.
        If the action space is discrete, ``action`` is assumed to already be in one-hot format.
        Action tensor is injected between each layer via concatenation to observation features
        following implementation from (Burda et al., Large-Scale Study of Curiosity-Driven Learning,
        https://arxiv.org/abs/1808.04355).

        Return format: ((next observation feature predictions, prediction error bonuses), FDM prediction error loss).
        Distinguishing between the transition-wise errors and fully reduced mean error allows for a modular
        use of various forward dynamics modules within the ICM.
        """
        x = torch.cat((obs_feat, action), dim=-1)
        x = self.fc_in(x)
        x = self.activ_in(x)
        for res_block in self.res_blocks:
            x = res_block((x, action))
        x = torch.cat((x, action), dim=-1)
        x = self.fc_out(x)
        pred_errors = torch.mean((x - next_obs_feat) ** 2, dim=-1)  # Maintains batch dimension
        fdm_loss = torch.mean(pred_errors)  # Reduces batch dimension
        return (x, pred_errors), fdm_loss


class ForwardDisagreementEnsemble(SelfSupervisedModule):  # TODO
    pass