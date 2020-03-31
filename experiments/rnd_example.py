
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.affinity import make_affinity

from intrinsic_rl.samplers.serial.sampler import IntrinsicSerialSampler
from intrinsic_rl.samplers.parallel.gpu.sampler import IntrinsicGpuSampler
from intrinsic_rl.algos.pg.intrinsic_ppo import IntrinsicPPO
from intrinsic_rl.agents.pg.rnd_agent import RndAtariFfAgent


def build_and_train(game="breakout", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = IntrinsicSerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "gpu":
        Sampler = IntrinsicGpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")

    sampler = Sampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=16,
        batch_B=8,
        obs_norm_steps=128*50,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    algo = IntrinsicPPO(int_rew_coeff=1., ext_rew_coeff=0.)

    rnd_model_kwargs = dict(
        channels=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[(4, 4), (2, 2), (1, 1)],
        hidden_sizes=[512],
        conv_nonlinearity=torch.nn.LeakyReLU)
    agent = RndAtariFfAgent(rnd_model_kwargs=rnd_model_kwargs)

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(50e6),
        log_interval_steps=int(1e2),
        affinity=affinity
    )

    config = dict(game=game)
    name = "breakout_" + game
    log_dir = "rnd_example"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='breakout')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'gpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel
    )