
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.logging import logger

from intrinsic_rl.samplers.buffer import build_intrinsic_samples_buffer
from intrinsic_rl.samplers.parallel.cpu.collectors import IntrinsicCpuResetCollector


class IntrinsicSerialSampler(SerialSampler):
    """
    Override of SerialSampler which uses IntrinsicCpuResetCollector.
    Also modifies initialize to construct samples buffer extended
    for intrinsic bonus agents, and handle initialization of observation
    normalization modules in the bonus model.
    """

    def __init__(self, *args, obs_norm_steps=0, CollectorCls=IntrinsicCpuResetCollector,
                 eval_CollectorCls=SerialEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
                         eval_CollectorCls=eval_CollectorCls, **kwargs)
        self.obs_norm_steps = obs_norm_steps

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            next_obs=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        """
        Override to call ``build_intrinsic_samples_buffer`` which handles allocating buffer
        space for sample components needed by intrinsic agents, such as intrinsic bootstrap value.
        Also handles initializing bonus model observation normalization.
        """
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        # Calls build_intrinsic_samples_buffer instead
        samples_pyt, samples_np, examples = build_intrinsic_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, next_obs=next_obs, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        # Run bonus model to initialize normalized obs model, if applicable
        if self.obs_norm_steps > 0:
            self.init_obs_norm(agent)

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples

    @torch.no_grad()
    def init_obs_norm(self, agent):
        """
        Initializes observation normalization parameters in intrinsic bonus model.
        Uses distinct environment for this purpose.
        """
        agent.set_norm_update(True)
        env = self.EnvCls(**self.env_kwargs)
        env.reset()
        logger.log(f"Sampler initializing bonus model observation normalization, steps: {self.obs_norm_steps}")
        for _ in range(self.obs_norm_steps):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            obs = torch.from_numpy(obs).to(device=agent.device)
            # Prepare observation, flattening channel dim (frame-stack) into batch dim for image input
            if len(obs.shape) == 3:  # (C, H, W)
                obs = obs.view((-1, 1, *obs.shape[1:]))
            agent.bonus_model.normalize_obs(obs)
            if done:
                env.reset()
