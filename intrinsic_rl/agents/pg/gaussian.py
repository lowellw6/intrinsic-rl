
import torch
from abc import ABC

from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.agents.base import AgentStep

from intrinsic_rl.agents.base import IntrinsicBonusAgent
from intrinsic_rl.agents.pg.base import IntAgentInfo, IntAgentValues


class IntrinsicGaussianPGAgent(IntrinsicBonusAgent, ABC):
    """
    Extends IntrinsicBonusAgent to support base models which produce intrinsic critic values
    for policies with continuous gaussian action output. Model calls are expected to produce separate
    critic values for intrinsic reward stream.
    """

    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, ev, iv = self.model(*model_inputs)
        return buffer_to((DistInfoStd(mean=mu, log_std=log_std), ev, iv), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, ev, iv = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = IntAgentInfo(dist_info=dist_info, ext_value=ev, int_value=iv)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, ev, iv = self.model(*model_inputs)
        ev, iv = buffer_to((ev, iv), device="cpu")
        return IntAgentValues(ext_value=ev, int_value=iv)
