
import torch

from rlpyt.envs.gym import make as gym_make
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.logging.logger import set_snapshot_gap

from intrinsic_rl.runners.minibatch_rl import MinibatchRlFlex
from intrinsic_rl.samplers.serial.sampler import IntrinsicSerialSampler
from intrinsic_rl.samplers.parallel.gpu.sampler import IntrinsicGpuSampler
from intrinsic_rl.algos.pg.rnd_algo import RndIntrinsicPPO
from intrinsic_rl.agents.pg.rnd_agent import RndMujocoFfAgent


def build_and_train(env="Ant-v2", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = IntrinsicSerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "gpu":
        Sampler = IntrinsicGpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")

    env_kwargs = dict(id=env)

    sampler = Sampler(
        EnvCls=gym_make,
        env_kwargs=env_kwargs,
        batch_T=128,
        batch_B=64,
        obs_norm_steps=0, #128*50,
        max_decorrelation_steps=0
    )

    algo = RndIntrinsicPPO(
        int_rew_coeff=1.,
        ext_rew_coeff=0.,
        ext_rew_clip=(-1, 1),
        minibatches=4,
        epochs=4,
        entropy_loss_coeff=0.001,
        learning_rate=0.0001,
        gae_lambda=0.95,
        discount=0.999,
        int_discount=0.99
    )

    rnd_model_kwargs = dict(
        hidden_sizes=[64, 64],
        output_size=10,
        nonlinearity=torch.nn.ReLU
    )
    base_model_kwargs = dict(  # Same front-end architecture as RND model, different fc kwarg name
        hidden_sizes=[64, 64],
        normalize_observation=True
    )
    agent = RndMujocoFfAgent(rnd_model_kwargs=rnd_model_kwargs, model_kwargs=base_model_kwargs)

    runner = MinibatchRlFlex(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(49152e4),  # this is 30k rollouts per environment at (T, B) = (128, 128)
        log_interval_steps=int(1e3),
        affinity=affinity
    )

    config = dict(game=env)
    name = "intrinsicPPO_" + env
    log_dir = "rnd_mujoco"
    set_snapshot_gap(1000)  # Save parameter checkpoint every 1000 training iterations
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="gap"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='Mujoco environment', default='Ant-v2')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'gpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        env=args.env,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel
    )
