
import multiprocessing as mp
from abc import ABC, abstractmethod

from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.worker import sampling_process
from rlpyt.utils.logging import logger

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
            next_obs=False,
            traj_info_kwargs=None,
            world_size=1,
            rank=0,
            worker_process=None,
    ):
        """
        Overrides initialize in ParallelSamplerBase to add next_observation support
        in _build_buffers and handle initializing observation normalization parameters.
        """
        n_envs_list = self._get_n_envs_list(affinity=affinity)
        self.n_worker = n_worker = len(n_envs_list)
        B = self.batch_spec.B
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        env = self.EnvCls(**self.env_kwargs)
        self._agent_init(agent, env, global_B=global_B,
                         env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value, next_obs)
        env.close()
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed, n_envs_list)

        target = sampling_process if worker_process is None else worker_process
        self.workers = [mp.Process(target=target,
                                   kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
                        for w_kwargs in workers_kwargs]
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait()  # Wait for workers ready (e.g. decorrelate).

        # Inserting observation normalization init run here
        if self.obs_norm_steps > 0:
            self.init_obs_norm()

        return examples  # e.g. In case useful to build replay buffer.

    @abstractmethod
    def init_obs_norm(self):
        """Steps agent to initialize observation normalization models."""
        pass

    def _build_buffers(self, env, bootstrap_value, next_obs):
        """Overrides method in ParallelSampler Base to use build_intrinsic_samples_buffer."""
        self.samples_pyt, self.samples_np, examples = build_intrinsic_samples_buffer(
            self.agent, env, self.batch_spec, bootstrap_value, next_obs,
            agent_shared=True, env_shared=True, subprocess=True)
        return examples
