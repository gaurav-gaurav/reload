import os
import random

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories, merge_trajectories
from d4rl_datasets import get_expert_traj
from evaluation import evaluate
from learner import Learner
import wandb
import uuid
import pyrallis

import jax
import jax.numpy as jnp
import optax
import chex
from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Optional, Callable
from tqdm.auto import trange
from flax.training.train_state import TrainState
# from networks import RND, Actor, EnsembleCritic, Alpha
from buffer import ReplayBuffer
from common_rnd import Metrics
# from running_moments import RunningMeanStd
import torch.nn.functional as F
import torch.nn as nn 
import torch
from rnd_torch import _RND
import torch.optim as optim
import sys


# sys.stdout = open(os.devnull, 'w')


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'walker2d-medium-replay-v2', 'Environment name.')
flags.DEFINE_string('save_dir', 'random_results/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e8), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_float('learning_rate', 3e-4, 'rnd learning_rate')
flags.DEFINE_integer('rnd_hidden_dim', 256, 'rnd_dims')
flags.DEFINE_integer('rnd_embedding_dim', 32, 'embedding_dim')
flags.DEFINE_string('rnd_mlp_type', "concat_first", 'rnd_concat_type')
flags.DEFINE_string('rnd_target_mlp_type' ,"concat_first", 'rnd_concat_type')
flags.DEFINE_boolean('rnd_switch_features', False, 'switch_features')
flags.DEFINE_integer('rnd_update_epoch', 100, ' number of updates')
flags.DEFINE_integer('training_seed', 0, 'train_seed')
# flags.DEFINE_integer('a', 10, 'alpha')
# flags.DEFINE_integer('b', 50, 'beta')

config_flags.DEFINE_config_file(
    'config',
	'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def train_predictor(obs, nxt_obs, RND, state_mean, state_std):
    optimizer = optim.Adam(RND.Predict_RND.parameters(), lr=1e-3)
    # obs = (obs - state_mean)/(state_std + 1e-8)
    # nxt_obs =  (nxt_obs - state_mean)/(state_std + 1e-8)

    ### for adroit results I used learning rate of 3e-4 and batchsize 1 
    print(len(obs))
    for i in range(200):
        idxs = np.random.randint(0, len(obs), size=32)
        obs = torch.as_tensor(obs[idxs], device=torch.cuda.current_device()).float()
        nxt_obs = torch.as_tensor(nxt_obs[idxs], device =torch.cuda.current_device()).float()

        predict_op = RND.Predict_RND(obs, nxt_obs)
        target_op = RND.Target_RND(obs, nxt_obs)
        loss = F.mse_loss(predict_op, target_op)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss_____________________:', loss)

def compute_reward_scale(traj):
    traj = traj.copy()
    def compute_returns(tr):
        return sum(tr['rewards'])

    traj.sort(key=compute_returns)
    scale = 1000/(compute_returns(traj[-1])-compute_returns(traj[0]))
    print('#############', scale)
    return scale


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    return env, dataset

def relabel_trajectories(expert_demo, rnd, dataset, state_mean, state_std):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    dataset = merge_trajectories(dataset, trajs, expert_demo, rnd,state_mean, state_std)
    scale = compute_reward_scale(trajs)
    return dataset, scale

def main(_):
    kwargs = dict(FLAGS.config)
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'state-state', FLAGS.env_name,
                                                str(FLAGS.seed)+'_'+str(kwargs['a'])+ str(kwargs['b'])),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    key = jax.random.PRNGKey(seed=FLAGS.training_seed)
    key, rnd_key, actor_key, critic_key, alpha_key = jax.random.split(key, 5)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    set_seed(FLAGS.seed, env)
    expert_demo = get_expert_traj(env)

    expert_obs = expert_demo['observations']
    expert_action = expert_demo['actions']
    expert_next_obs = expert_demo['next_observations']#[1:]
    # expert_next_obs.append(expert_demo['next_observations'][-1])

    print('################',kwargs['a'],kwargs['b'])

    # init_state = expert_obs[0][None, ...]
    # init_action = expert_next_obs[0][None, ...]
    RND = _RND(env.observation_space, env.action_space)
    # buffer_rnd = rnd_buffer(expert_demo)

    # kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    state_mean = dataset.observations.mean(0)
    state_std = dataset.observations.std(0)
    train_predictor(np.asarray(expert_demo['observations']), np.asarray(expert_demo['next_observations']), RND, state_mean, state_std)
    dataset, reward_scale = relabel_trajectories(expert_demo, RND, dataset, state_mean, state_std)
    if 'antmaze' in FLAGS.env_name:
    	reward_bias=-2
    else:
    	reward_bias = 0
    reward_mean = dataset.rewards.mean(0)
    reward_std = dataset.rewards.std(0)
    dataset.rewards = np.asarray([reward*reward_scale + reward_bias for reward in dataset.rewards])
    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, 'state-state', FLAGS.env_name, 
                        f"{FLAGS.seed}_{kwargs['a']}_1_10_{kwargs['b']}.txt"),
           				eval_returns,
           				fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
