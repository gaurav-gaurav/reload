import d4rl
import gym
import numpy as np
import tqdm
from collections import defaultdict

from dataset import Dataset

def qlearning_dataset_with_timeouts(env,
                                    dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
  if dataset is None:
    dataset = env.get_dataset(**kwargs)

  N = dataset['rewards'].shape[0]
  obs_ = []
  next_obs_ = []
  action_ = []
  reward_ = []
  done_ = []
  realdone_ = []
  if "infos/goal" in dataset:
    if not disable_goal:
      dataset["observations"] = np.concatenate(
          [dataset["observations"], dataset['infos/goal']], axis=1)
    else:
      pass
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)

  episode_step = 0
  for i in range(N - 1):
    obs = dataset['observations'][i]
    new_obs = dataset['observations'][i + 1]
    action = dataset['actions'][i]
    reward = dataset['rewards'][i]
    done_bool = bool(dataset['terminals'][i])
    realdone_bool = bool(dataset['terminals'][i])
    if "infos/goal" in dataset:
      final_timestep = True if (dataset['infos/goal'][i] !=
                                dataset['infos/goal'][i + 1]).any() else False
    else:
      final_timestep = dataset['timeouts'][i]

    if i < N - 1:
      done_bool += final_timestep

    if (not terminate_on_end) and final_timestep:
      # Skip this transition and don't apply terminals on the last step of an episode
      episode_step = 0
      continue
    if done_bool or final_timestep:
      episode_step = 0

    obs_.append(obs)
    next_obs_.append(new_obs)
    action_.append(action)
    reward_.append(reward)
    done_.append(done_bool)
    realdone_.append(realdone_bool)
    episode_step += 1

  return {
      'observations': np.array(obs_),
      'actions': np.array(action_),
      'next_observations': np.array(next_obs_),
      'rewards': np.array(reward_)[:],
      'terminals': np.array(done_)[:],
      'realterminals': np.array(realdone_)[:],
  }
  
class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, fix_antmaze_timeout=True, clip_to_eps: bool = True, eps: float = 1e-5):
        if "antmaze" in env.unwrapped.spec.id and fix_antmaze_timeout:
            dataset_dict = qlearning_dataset_with_timeouts(env)
        else:
            dataset_dict = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True
        if 'realterminals' in dataset_dict:
            # We updated terminals in the dataset, but continue using
            # the old terminals for consistency with original IQL.
            dataset_dict['masks'] = 1.0 - dataset_dict['realterminals'].astype(np.float32)
            del dataset_dict["realterminals"]
        else:
            dataset_dict['masks'] = 1.0 - dataset_dict['terminals'].astype(np.float32)
            del dataset_dict["terminals"]

        # dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)

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



def concat_dicts(dict_list):
    result = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            result[k].append(v)
    return dict(result)

def get_expert_traj(env):
    expert_ds = D4RLDataset(env)
    offline_traj = split_into_trajectories(expert_ds.dataset_dict['observations'].astype(np.float32),
        expert_ds.dataset_dict['actions'].astype(np.float32), 
        expert_ds.dataset_dict['rewards'].astype(np.float32),
        expert_ds.dataset_dict['masks'],
        expert_ds.dataset_dict['dones'].astype(np.float32),
        expert_ds.dataset_dict['next_observations'].astype(np.float32))
    # offline_traj is the list of trajectories with each trajectory being a dictionary
    if "antmaze" in env.unwrapped.spec.id:
        returns = [sum(traj['rewards']) / (1e-4 + np.linalg.norm(traj['observations'][0][:2])) for traj in offline_traj]
    else:
        returns = [sum(traj['rewards']) for traj in offline_traj]  # Sum of rewards for each trajectory
    idx_max_return = np.argsort(returns)[-1:]
    # demo_returns = returns[idx_max_return]
    # expert_demo = offline_traj[idx_max_return]
    # print(f"demo returns {demo_returns}, mean {np.mean(demo_returns)}")
    expert_demo = [offline_traj[i] for i in idx_max_return]
    expert_demo = concat_dicts(expert_demo)
    expert_demo = {k: np.concatenate(v, axis=0) for k, v in expert_demo.items()}

    del expert_ds
    return expert_demo 
