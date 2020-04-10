

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="breakout", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")

    env_kwargs = dict(game=game, repeat_action_probability=0.25, horizon=int(18e3))

    sampler = Sampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=env_kwargs,
        batch_T=128,
        batch_B=1,
        max_decorrelation_steps=0
    )

    algo = PPO(
        minibatches=4,
        epochs=4,
        entropy_loss_coeff=0.001,
        learning_rate=0.0001,
        gae_lambda=0.95,
        discount=0.999
    )

    base_model_kwargs = dict(  # Same front-end architecture as RND model, different fc kwarg name
        channels=[32, 64, 64],
        kernel_sizes=[8, 4, 4],
        strides=[(4, 4), (2, 2), (1, 1)],
        paddings=[0, 0, 0],
        fc_sizes=[512]
        # Automatically applies nonlinearity=torch.nn.ReLU in this case,
        # but can't specify due to rlpyt limitations
    )
    agent = AtariFfAgent(model_kwargs=base_model_kwargs)

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(49152e4),  # this is 30k rollouts per environment at (T, B) = (128, 128)
        log_interval_steps=int(1e3),
        affinity=affinity
    )

    config = dict(game=game)
    name = "ppo_" + game
    log_dir = "baseline"
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
    )