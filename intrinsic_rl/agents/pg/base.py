
import torch
from abc import ABC

from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.categorical import DistInfo
from rlpyt.agents.base import AgentStep
from rlpyt.utils.collections import namedarraytuple

from intrinsic_rl.agents.base import IntrinsicBonusAgent

IntAgentInfo = namedarraytuple("IntAgentInfo", ["dist_info", "ext_value", "int_value"])
IntAgentValues = namedarraytuple("IntAgentValues", ["ext_value", "int_value"])


class IntrinsicPolicyGradientAgent(IntrinsicBonusAgent, ABC):
    """
    Extends IntrinsicBonusAgent to support base models which produce intrinsic critic values.
    Model calls are expected to produce separate critic values for intrinsic reward stream.
    """

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.format_actions(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi, ext_value, int_value = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), ext_value, int_value), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.format_actions(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        pi, ext_value, int_value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = IntAgentInfo(dist_info=dist_info, ext_value=ext_value, int_value=int_value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.format_actions(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        _, ext_value, int_value = self.model(*model_inputs)
        ext_value, int_value = buffer_to((ext_value, int_value), device="cpu")
        return IntAgentValues(ext_value=ext_value, int_value=int_value)
