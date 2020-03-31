
from abc import ABC, abstractmethod

from rlpyt.samplers.parallel.base import ParallelSamplerBase

from intrinsic_rl.samplers.buffer import build_intrinsic_samples_buffer


class IntrinsicParallelSamplerBase(ParallelSamplerBase, ABC):
    """ParallelSamplerBase which supports intrinsic agent needs, such as providing additional buffer contents."""

    gpu = False

    def __init__(self, *args, obs_norm_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_norm_steps = obs_norm_steps

    def initialize(
            self,
            agent,
            affinity,
            seed,
            bootstrap_value=False,
            traj_info_kwargs=None,
            world_size=1,
            rank=0,
            worker_process=None,
            ):
        """###"""
        examples = super().initialize(
            agent,
            affinity,
            seed,
            bootstrap_value=bootstrap_value,
            traj_info_kwargs=traj_info_kwargs,
            world_size=world_size,
            rank=rank,
            worker_process=worker_process,
        )
        # Inserting observation normalization init run here
        if self.obs_norm_steps > 0:
            self.init_obs_norm()
        return examples

    @abstractmethod
    def init_obs_norm(self):
        """Steps agent to initialize observation normalization models."""
        pass

    def _build_buffers(self, env, bootstrap_value):
        """Overrides method in ParallelSampler Base to use build_intrinsic_samples_buffer."""
        self.samples_pyt, self.samples_np, examples = build_intrinsic_samples_buffer(
            self.agent, env, self.batch_spec, bootstrap_value,
            agent_shared=True, env_shared=True, subprocess=True)
        return examples
