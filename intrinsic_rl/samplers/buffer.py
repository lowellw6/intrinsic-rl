
import multiprocessing as mp

from rlpyt.utils.collections import namedarraytuple
from rlpyt.samplers.buffer import get_example_outputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer
from rlpyt.samplers.collections import Samples, AgentSamples, EnvSamples

IntAgentSamplesBsv = namedarraytuple("IntAgentSamplesBsv",
                                     ["action", "prev_action", "agent_info", "bootstrap_value", "int_bootstrap_value"])


def build_intrinsic_samples_buffer(agent, env, batch_spec, bootstrap_value=False,
                                   agent_shared=True, env_shared=True, subprocess=True, examples=None):
    """
    Replaces ``build_samples_buffer`` to add additional buffer space for intrinsic bonus agents.
    If bootstrap_value=True, also adds space for int_bootstrap_value from intrinsic value head.
    """
    if examples is None:
        if subprocess:
            mgr = mp.Manager()
            examples = mgr.dict()  # Examples pickled back to master.
            w = mp.Process(target=get_example_outputs,
                args=(agent, env, examples, subprocess))
            w.start()
            w.join()
        else:
            examples = dict()
            get_example_outputs(agent, env, examples)

    T, B = batch_spec
    all_action = buffer_from_example(examples["action"], (T + 1, B), agent_shared)
    action = all_action[1:]
    prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    agent_info = buffer_from_example(examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:  # Added buffer space for intrinsic bootstrap value
        bv = buffer_from_example(examples["agent_info"].ext_value, (1, B), agent_shared)
        int_bv = buffer_from_example(examples["agent_info"].int_value, (1, B), agent_shared)
        agent_buffer = IntAgentSamplesBsv(*agent_buffer, bootstrap_value=bv, int_bootstrap_value=int_bv)

    observation = buffer_from_example(examples["observation"], (T, B), env_shared)
    all_reward = buffer_from_example(examples["reward"], (T + 1, B), env_shared)
    reward = all_reward[1:]
    prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(examples["done"], (T, B), env_shared)
    env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent=agent_buffer, env=env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np, examples
