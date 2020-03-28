
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer

from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector


class IntrinsicCpuResetCollector(CpuResetCollector):
    """
    CpuResetCollector extended to collect sample components specific
    to intrinsic bonus agents, such as intrinsic bootstrap values.
    """

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                    env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        # Modified to include int_bootstrap_value
        if "bootstrap_value" in agent_buf:
            bootstraps = self.agent.value(obs_pyt, act_pyt, rew_pyt)
            agent_buf.bootstrap_value[:] = bootstraps.ext_value
            agent_buf.int_bootstrap_value[:] = bootstraps.int_value

        return AgentInputs(observation, action, reward), traj_infos, completed_infos