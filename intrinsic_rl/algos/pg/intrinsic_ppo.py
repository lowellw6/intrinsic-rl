
import torch
from abc import ABC

from rlpyt.algos.pg.base import OptInfo
from rlpyt.algos.pg.ppo import PPO
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.collections import namedarraytuple

from intrinsic_rl.algos.pg.base import IntrinsicPolicyGradientAlgo

ActorInputs = namedarraytuple("ActorInputs", ["observation", "prev_action", "prev_reward"])
LossInputs = namedarraytuple("LossInputs",
    ["actor_inputs", "action", "int_bootstrap_val", "ext_return", "ext_adv", "valid", "old_dist_info"])


class IntrinsicPPO(PPO, IntrinsicPolicyGradientAlgo, ABC):
    """
    Abstract base class for PPO using an intrinsic bonus model.
    Must override abstract method ``extract_bonus_inputs`` based on
    specific intrinsic bonus model / algorithm to be used.
    """

    def __init__(self,
                 int_discount=0.99,  # Separate discount factor for intrinsic reward stream
                 int_rew_coeff=1.,
                 ext_rew_coeff=0.,
                 bonus_loss_coeff=1.,
                 entropy_loss_coeff=0.,  # Default is to discard policy entropy
                 **kwargs):
        save__init__args(locals())
        super().__init__(entropy_loss_coeff=entropy_loss_coeff, **kwargs)

    def optimize_agent(self, itr, samples):
        """
        Override to provide additional flexibility in what enters the combined_loss function.
        """
        recurrent = self.agent.recurrent
        actor_inputs = ActorInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        actor_inputs = buffer_to(actor_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(actor_inputs.observation)
        ext_return, ext_adv, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            actor_inputs=actor_inputs,
            action=samples.agent.action,
            int_bootstrap_val=samples.agent.int_bootstrap_value,  # Additional bootstrap val for intrinsic reward stream
            ext_return=ext_return,
            ext_adv=ext_adv,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                # Combined loss produces single loss for both actor and bonus model
                loss, entropy, perplexity = self.combined_loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def combined_loss(self, agent_inputs, action, int_bootstrap_val, ext_return, ext_adv, valid, old_dist_info,
            init_rnn_state=None):
        """
        Alternative to ``loss`` in PPO.
        This functions runs ``bonus_call``, performing a forward pass of the intrinsic bonus model
        and producing a combined reward/advantage stream, and then a combined loss.
        """
        # Run base actor critic model
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, ext_value, int_value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, ext_value, int_value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        # Extract bonus model inputs and call bonus model, generating intrinsic rewards
        bonus_model_inputs = self.extract_bonus_inputs(agent_inputs, action)
        int_rew, bonus_loss = self.agent.bonus_call(bonus_model_inputs)
        bonus_loss *= self.bonus_loss_coeff

        # Process intrinsic reward stream, and produce combined advantages
        int_return, int_adv = self.process_intrinsic_returns(int_rew, int_value.detach(), int_bootstrap_val)
        advantage = self.ext_rew_coeff * ext_adv + self.int_rew_coeff * int_adv

        # Construct PPO loss
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        ext_value_error = 0.5 * (ext_value - ext_return) ** 2
        int_value_error = 0.5 * (int_value - int_return) ** 2
        value_loss = self.value_loss_coeff * (valid_mean(ext_value_error, valid) + int_value_error.mean())

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss + bonus_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity