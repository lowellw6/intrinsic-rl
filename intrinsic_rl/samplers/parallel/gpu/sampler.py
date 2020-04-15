
import numpy as np
import torch

from rlpyt.samplers.parallel.gpu.sampler import GpuSamplerBase
from rlpyt.utils.synchronize import drain_queue
from rlpyt.utils.logging import logger

from intrinsic_rl.samplers.parallel.base import IntrinsicParallelSamplerBase
from intrinsic_rl.samplers.parallel.gpu.collectors import IntrinsicGpuResetCollector
from intrinsic_rl.samplers.parallel.gpu.action_server import IntrinsicActionServer


class IntrinsicGpuSamplerBase(GpuSamplerBase, IntrinsicParallelSamplerBase):
    """GpuSamplerBase which provides routines for initializing observation normalization models."""
    gpu = True

    def __init__(self, *args, CollectorCls=IntrinsicGpuResetCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls, **kwargs)

    @torch.no_grad()
    def init_obs_norm(self):
        """
        Initializes observation normalization parameters in intrinsic bonus model.
        Agent base network is not stepped, rather the action space is sampled randomly
        to exercise the bonus model obs norm module. This will run for at least as many
        steps specified in self.obs_norm_steps.
        """
        logger.log(f"Sampler initializing bonus model observation normalization, steps: {self.obs_norm_steps}")
        action_space = self.EnvCls(**self.env_kwargs).action_space
        world_batch_size = self.batch_size * self.world_size
        from math import ceil
        for _ in range(ceil(self.obs_norm_steps / world_batch_size)):
            self.ctrl.barrier_in.wait()
            self.run_obs_norm(action_space)
            self.ctrl.barrier_out.wait()
            drain_queue(self.traj_infos_queue)

    def run_obs_norm(self, action_space):
        """
        Exercises observation normalization model within bonus model. Used in initialization.
        Follows Action Server ``serve_actions`` pattern to make use of parallel workers.
        """
        obs_ready, act_ready = self.sync.obs_ready, self.sync.act_ready
        step_np, agent_inputs = self.step_buffer_np, self.agent_inputs

        for t in range(self.batch_spec.T):
            for b in obs_ready:
                b.acquire()  # Workers written obs and rew, first prev_act.
                # assert not b.acquire(block=False)  # Debug check.
            if self.mid_batch_reset and np.any(step_np.done):
                for b_reset in np.where(step_np.done)[0]:
                    step_np.action[b_reset] = 0  # Null prev_action into agent.
                    step_np.reward[b_reset] = 0  # Null prev_reward into agent.

            # Prepare observation, flattening channel dim (frame-stack) into batch dim for image input
            observation = agent_inputs.observation
            if len(observation.shape) == 4:  # (B, C, H, W)
                observation = observation.view((-1, 1, *observation.shape[2:]))

            # exercise observation normalization model
            self.agent.set_norm_update(True)
            self.agent.bonus_model.normalize_obs(observation)

            # get random actions for workers
            action = np.empty_like(step_np.action)
            for B in range(self.batch_spec.B):
                action[B] = action_space.sample()
            step_np.action[:] = action  # Worker applies to env.

            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()  # Signal to worker.

        for b in obs_ready:
            b.acquire()
            assert not b.acquire(block=False)  # Debug check.
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
            # step_np.done[:] = False  # Worker resets at start of next.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.


class IntrinsicGpuSampler(IntrinsicActionServer, IntrinsicGpuSamplerBase):
    pass
