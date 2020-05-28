
import torch
from abc import ABC

from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.collections import namedarraytuple

from intrinsic_rl.algos.pg.base import IntrinsicPolicyGradientAlgo

import cv2  ###

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "next_obs", "ext_return", "ext_adv", "int_return", "int_adv", "valid", "old_dist_info"])
OptInfo = namedarraytuple("OptInfo",
    ["loss", "policyLoss", "valueLoss", "entropyLoss", "bonusLoss",
     "extrinsicValue", "intrinsicValue",
     "intrinsicReward", "discountedIntrinsicReturn",
     "gradNorm", "entropy", "perplexity",
     "meanObsRmsModel", "varObsRmsModel",
     "meanIntRetRmsModel", "varIntRetRmsModel"])


class IntrinsicPPO(PPO, IntrinsicPolicyGradientAlgo, ABC):
    """
    Abstract base class for PPO using an intrinsic bonus model.
    Must override abstract method ``extract_bonus_inputs`` based on
    specific intrinsic bonus model / algorithm to be used.
    """

    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(self,
                 int_discount=0.99,  # Separate discount factor for intrinsic reward stream
                 int_rew_coeff=1.,
                 ext_rew_coeff=0.,
                 bonus_loss_coeff=1.,
                 entropy_loss_coeff=0.,  # Default is to discard policy entropy
                 ext_rew_clip=None,   # Clip range for extrinsic rewards as tuple (min, max)
                 **kwargs):
        save__init__args(locals())
        super().__init__(entropy_loss_coeff=entropy_loss_coeff, **kwargs)

    def optimize_agent(self, itr, samples):
        """
        Override to provide additional flexibility in what enters the combined_loss function.
        """
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)

        # Process extrinsic returns and advantages
        ext_rew, done, ext_val, ext_bv = (samples.env.reward, samples.env.done,
                                          samples.agent.agent_info.ext_value, samples.agent.bootstrap_value)
        done = done.type(ext_rew.dtype)
        if self.ext_rew_clip:  # Clip extrinsic reward is specified
            rew_min, rew_max = self.ext_rew_clip
            ext_rew = ext_rew.clamp(rew_min, rew_max)
        ext_return, ext_adv, valid = self.process_extrinsic_returns(ext_rew, done, ext_val, ext_bv)

        # Gather next observations, or fill with dummy placeholder (current obs)
        # Note the agent decides what it extracts and uses as input to its model,
        # so the dummy tensor scenario will have no effect
        next_obs = samples.env.next_observation if "next_observation" in samples.env else samples.env.observation

        # First call to bonus model, generates intrinsic rewards for samples batch
        # [T, B] leading dims are flattened, and the resulting returns are unflattened
        batch_shape = samples.env.observation.shape[:2]
        bonus_model_inputs = self.agent.extract_bonus_inputs(
            observation=samples.env.observation.flatten(end_dim=1),
            next_observation=next_obs.flatten(end_dim=1),  # May be same as observation (dummy placeholder) if algo set next_obs=False
            action=samples.agent.action.flatten(end_dim=1)
        )
        self.agent.set_norm_update(True)  # Bonus model will update any normalization models where applicable
        int_rew, _ = self.agent.bonus_call(bonus_model_inputs)
        int_rew = int_rew.view(batch_shape)

        # Process intrinsic returns and advantages (updating intrinsic reward normalization model, if applicable)
        int_val, int_bv = samples.agent.agent_info.int_value, samples.agent.int_bootstrap_value
        int_return, int_adv = self.process_intrinsic_returns(int_rew, int_val, int_bv)

        # Avoid repeating any norm updates on same data in subsequent loss forward calls
        self.agent.set_norm_update(False)

        # Add front-processed optimizer data to logging buffer
        # Flattened to match elsewhere, though the ultimate statistics summarize over all dims anyway
        opt_info.extrinsicValue.extend(ext_val.flatten().tolist())
        opt_info.intrinsicValue.extend(int_val.flatten().tolist())
        opt_info.intrinsicReward.extend(int_rew.flatten().tolist())
        opt_info.discountedIntrinsicReturn.extend(int_return.flatten().tolist())
        opt_info.meanObsRmsModel.extend(self.agent.bonus_model.obs_rms.mean.flatten().tolist())
        opt_info.varObsRmsModel.extend(self.agent.bonus_model.obs_rms.var.flatten().tolist())
        opt_info.meanIntRetRmsModel.extend(self.agent.bonus_model.int_ret_rms.mean.flatten().tolist())
        opt_info.varIntRetRmsModel.extend(self.agent.bonus_model.int_ret_rms.var.flatten().tolist())

        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            next_obs=next_obs,
            ext_return=ext_return,
            ext_adv=ext_adv,
            int_return=int_return,
            int_adv=int_adv,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        T, B = samples.env.reward.shape[:2]
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
                loss, entropy, perplexity, pi_loss, value_loss, entropy_loss, bonus_loss = \
                    self.combined_loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.policyLoss.append(pi_loss.item())
                opt_info.valueLoss.append(value_loss.item())
                opt_info.entropyLoss.append(entropy_loss.item())
                opt_info.bonusLoss.append(bonus_loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def combined_loss(self, agent_inputs, action, next_obs, ext_return, ext_adv, int_return, int_adv,
                      valid, old_dist_info, init_rnn_state=None):
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

        # Second call to bonus model, generates self-supervised bonus model loss
        # Leading batch dims have already been flattened after entering minibatch
        bonus_model_inputs = self.agent.extract_bonus_inputs(
            observation=agent_inputs.observation,
            next_observation=next_obs,  # May be same as observation (dummy placeholder) if algo set next_obs=False
            action=action
        )
        _, bonus_loss = self.agent.bonus_call(bonus_model_inputs)
        bonus_loss *= self.bonus_loss_coeff

        # Fuse reward streams by producing combined advantages
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
        return loss, entropy, perplexity, pi_loss, value_loss, entropy_loss, bonus_loss
