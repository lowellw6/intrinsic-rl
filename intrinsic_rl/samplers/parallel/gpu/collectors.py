
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector


class IntrinsicGpuResetCollector(GpuResetCollector):
    """
    GpuResetCollector extended to collect sample components specific
    to intrinsic bonus agents, such as next observations, if applicable.
    """

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """
        Identical to ``collect_batch`` in GpuResetCollector except
        gathers next_observation at end, if present in buffer.
        """
        act_ready, obs_ready = self.sync.act_ready, self.sync.obs_ready
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        env_buf.prev_reward[0] = step.reward
        obs_ready.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_ready.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = step.action  # OPTIONAL BY SERVER
            env_buf.reward[t] = step.reward
            env_buf.done[t] = step.done
            if step.agent_info:
                agent_buf.agent_info[t] = step.agent_info  # OPTIONAL BY SERVER
            if "next_observation" in env_buf:  # Modified to include next obs
                env_buf.next_observation[t] = step.observation
            obs_ready.release()  # Ready for server to use/write step buffer.

        return None, traj_infos, completed_infos
