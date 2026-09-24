import collections
from typing import Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import ott
from ott.geometry import pointcloud
from ott.geometry import costs
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import tqdm
import ot
# from flax.training.train_state import TrainState
# from running_moments import RunningMeanStd
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn 
from configs import mujoco_config
config = mujoco_config.get_config()


mse_loss = nn.MSELoss(reduction='none')

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

def cosine_distance(x, y):
    C = np.dot(x, y.T)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.dot(x_norm, y_norm.T)
    C = 1 - C / norms
    return C

def euclidean_distance(x, y):
    """Returns the matrix of Euclidean distances."""
    x_col = np.expand_dims(x, axis=1)
    y_lin = np.expand_dims(y, axis=0)
    c = np.sqrt(np.sum(np.abs(x_col - y_lin) ** 2, axis=2))
    return c

def optimal_transport_plan(X, Y, cost_matrix, method='sinkhorn', niter=500, epsilon=0.01):
    X_pot = np.ones(X.shape[0]) / X.shape[0]
    Y_pot = np.ones(Y.shape[0]) / Y.shape[0]
    c_m = cost_matrix.copy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    return transport_plan

def compute_reward_scale(traj):
    traj = traj.copy()
    def compute_returns(tr):
        return sum(tr['rewards'])
    
    traj.sort(key=compute_returns)
    scale = (compute_returns(traj[-1]) - compute_returns(traj[0]))
    return scale

def compute_reward(exp, obs, mask):
    obs, exp = np.asarray(obs), np.asarray(exp)
    cost_matrix = cosine_distance(obs, exp)  # Get cost matrix for samples using critic network.
    transport_plan = optimal_transport_plan(obs, exp, cost_matrix, method='sinkhorn', niter=200)  # Getting optimal coupling
    ot_rewards = -np.diag(np.dot(transport_plan, cost_matrix.T))
    ot_rewards = 10 * np.exp(ot_rewards* 1)
    ot_rewards = np.where(mask, ot_rewards, 0)
    # print(ot_rewards.shape)
    return ot_rewards


def get_rnd_reward(RND, obs, nxt_obs, mask, state_mean, state_std):
    # obs = (obs - state_mean)/(state_std + 1e-8)
    # nxt_obs =  (nxt_obs - state_mean)/(state_std + 1e-8)
    # nxt_obs[:-1] = nxt_obs[1:]
    obs = torch.as_tensor(obs, device=torch.cuda.current_device()).float()
    nxt_obs = torch.as_tensor(nxt_obs, device=torch.cuda.current_device()).float()
    predict_op = RND.Predict_RND(obs, nxt_obs)
    target_op = RND.Target_RND(obs, nxt_obs)
    loss = -torch.mean(mse_loss(predict_op, target_op),1)
    # cost_matrix = cosine_distance(predict_op, target_op)  # Get cost matrix for samples using critic network.
    # transport_plan = optimal_transport_plan(predict_op, target_op, cost_matrix, method='sinkhorn',niter=200).float()  # Getting optimal coupling
    # ot_rewards = -torch.diag(torch.mm(transport_plan,
    #                 cost_matrix.T))
    # ot_rewards = config.a*torch.exp(ot_rewards*config.b)
    loss = config.a*torch.exp(loss*config.b)
    loss = loss.detach().cpu().numpy()
    loss = np.where(mask, loss, 0)
    # loss = compute_reward(target_op, predict_op, mask)
    return loss

def split_into_trajectories(observations, actions, rewards, masks, dones_float, next_observations):
    traj = []
    current_trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'masks': [],
        'dones_float': [],
        'next_observations': []
    }
    for i in tqdm.tqdm(range(len(observations))):
        current_trajectory['observations'].append(observations[i])
        current_trajectory['actions'].append(actions[i])
        current_trajectory['rewards'].append(rewards[i])
        current_trajectory['dones_float'].append(dones_float[i])
        current_trajectory['masks'].append(masks[i])
        current_trajectory['next_observations'].append(next_observations[i])

        if dones_float[i] == 1.0:
            traj.append(current_trajectory)  
            current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones_float': [],
                'masks': [],
                'next_observations': []
            }
    return traj

def merge_trajectories(dataset, trajs, expert_demo, rnd, state_mean, state_std):
    dataset.observations
    expert_demo_observations = expert_demo['observations']
    recalibrated_offline_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones_float': [],
        'masks': [],
        'next_observations': []
        }
    for traj in trajs:
        # computed_rewards = compute_reward(expert_demo_observations, traj['observations'], traj['masks'])
        computed_rewards = get_rnd_reward(rnd, traj['observations'], traj['next_observations'], traj['masks'], state_mean, state_std)
        traj['rewards'] = computed_rewards
    for traj in trajs:
        recalibrated_offline_data['observations'].extend(traj['observations'])
        recalibrated_offline_data['actions'].extend(traj['actions'])
        recalibrated_offline_data['rewards'].extend(traj['rewards'])
        recalibrated_offline_data['dones_float'].extend(traj['dones_float'])
        recalibrated_offline_data['masks'].extend(traj['masks'])
        recalibrated_offline_data['next_observations'].extend(traj['next_observations'])

    dataset.observations = np.asarray(recalibrated_offline_data['observations'])
    dataset.actions = np.asarray(recalibrated_offline_data['actions'])
    dataset.rewards = np.asarray(recalibrated_offline_data['rewards'])
    dataset.dones = np.asarray(recalibrated_offline_data['dones_float'])
    dataset.masks = np.asarray(recalibrated_offline_data['masks'])
    dataset.next_observations = np.asarray(recalibrated_offline_data['next_observations'])
    return dataset

class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
