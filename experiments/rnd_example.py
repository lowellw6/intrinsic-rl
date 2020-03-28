
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

from intrinsic_rl.samplers.serial.sampler import IntrinsicSerialSampler
from intrinsic_rl.algos.pg.intrinsic_ppo import IntrinsicPPO
from intrinsic_rl.agents.pg.rnd_agent import RndAtariFfAgent


def build_and_train(game="breakout", run_ID=0, cuda_idx=None):
    sampler = IntrinsicSerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        obs_norm_steps=128*50,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    algo = IntrinsicPPO(int_rew_coeff=0., ext_rew_coeff=1.)

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
        affinity=dict(cuda_idx=cuda_idx),
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
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )