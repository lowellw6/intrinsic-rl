
import numpy as np

from rlpyt.samplers.parallel.gpu.action_server import ActionServer


class IntrinsicActionServer(ActionServer):
    """Action Server which supports sampling of agent steps for intrinsic agents."""

    def serve_actions(self, itr):
        """Overrides to include sampling of intrinsic bootstrap value."""
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
                    self.agent.reset_one(idx=b_reset)
            action, agent_info = self.agent.step(*agent_inputs)
            step_np.action[:] = action  # Worker applies to env.
            step_np.agent_info[:] = agent_info  # Worker sends to traj_info.
            for w in act_ready:
                # assert not w.acquire(block=False)  # Debug check.
                w.release()  # Signal to worker.

        for b in obs_ready:
            b.acquire()
            assert not b.acquire(block=False)  # Debug check.
        if "bootstrap_value" in self.samples_np.agent:  # Modified to include intrinsic bootstrap
            bootstraps = self.agent.value(*agent_inputs)
            self.samples_np.agent.bootstrap_value[:] = bootstraps.ext_value
            self.samples_np.agent.int_bootstrap_value[:] = bootstraps.int_value
        if np.any(step_np.done):  # Reset at end of batch; ready for next.
            for b_reset in np.where(step_np.done)[0]:
                step_np.action[b_reset] = 0  # Null prev_action into agent.
                step_np.reward[b_reset] = 0  # Null prev_reward into agent.
                self.agent.reset_one(idx=b_reset)
            # step_np.done[:] = False  # Worker resets at start of next.
        for w in act_ready:
            assert not w.acquire(block=False)  # Debug check.