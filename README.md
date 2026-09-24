# ReLOAD: Self-Distilled Rewards for Offline Reinforcement Learning

Reference implementation of **ReLOAD**, which learns from unlabelled offline
trajectories by generating its own reward signal with Random Network
Distillation — removing the need for reward annotation.

> Gaurav Chaudhary, Laxmidhar Behera.
> *From Novelty to Imitation: Self-Distilled Rewards for Offline Reinforcement Learning.*
> Transactions on Machine Learning Research (TMLR), 2025.
> [OpenReview](https://openreview.net/forum?id=F5K94JI2Jb)

## Method

Offline RL normally needs reward labels, which are expensive to engineer and
often impossible to recover after the fact. ReLOAD instead trains an RND
predictor against a frozen random target on expert trajectories. The predictor's
error is low on states resembling the demonstrations and high elsewhere, so the
negated error serves as a dense reward derived entirely from the data. An IQL
agent is then trained on those self-distilled rewards.

| File | Role |
|---|---|
| `rnd_torch.py` | **The contribution.** RND target/predictor pair and the self-distilled reward. |
| `d4rl_datasets.py` | Expert trajectory extraction from D4RL. |
| `dataset.py` | Dataset container used by the above. |
| `learner.py` | IQL learner, adapted to consume the generated rewards. |
| `train_offline.py` | Training entry point. |
| `run_process.py` | Sweep launcher. |

## Install

```bash
git clone https://github.com/gaurav-gaurav/reload.git
cd reload
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# JAX with GPU support — match your CUDA version
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

[D4RL](https://github.com/Farama-Foundation/D4RL) needs MuJoCo installed; see
its instructions. Note the codebase is primarily JAX, with the RND module in
PyTorch, so both are required.

## Usage

```bash
# Locomotion
python train_offline.py --env_name=walker2d-medium-replay-v2 \
    --config=configs/mujoco_config.py

# AntMaze
python train_offline.py --env_name=antmaze-large-play-v0 \
    --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000

# Adroit / Kitchen
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

### RND arguments

| Flag | Default | Meaning |
|---|---|---|
| `--learning_rate` | `3e-4` | RND predictor learning rate |
| `--rnd_hidden_dim` | `256` | RND hidden width |
| `--rnd_embedding_dim` | `32` | RND embedding width |
| `--rnd_mlp_type` | `concat_first` | How state and action are combined in the predictor |
| `--rnd_target_mlp_type` | `concat_first` | Same, for the frozen target |
| `--rnd_switch_features` | `False` | Swap the feature order |
| `--rnd_update_epoch` | `100` | RND pretraining epochs before agent training |
| `--batch_size` | `256` | Mini-batch size |
| `--max_steps` | `1e8` | Training steps |
| `--seed` | `42` | Random seed |

`python train_offline.py --help` lists all flags.

## Attribution

This repository is a derivative work and its licensing reflects that — see
[LICENSE](LICENSE) for the full breakdown.

**Built on IQL (MIT).** The offline RL agent is the reference implementation of
[Implicit Q-Learning](https://github.com/ikostrikov/implicit_q_learning) by
Kostrikov, Nair and Levine, modified here to train on self-distilled rewards.
Their notice is preserved in [LICENSE.iql](LICENSE.iql).

**Two files from sac-rnd (Apache 2.0).** `buffer.py` and `common_rnd.py` are
reproduced unmodified from
[sac-rnd](https://github.com/tinkoff-ai/sac-rnd) (Nikulin et al., 2023); see
[LICENSE.sac-rnd](LICENSE.sac-rnd).

**Ours:** `rnd_torch.py`, `d4rl_datasets.py`, `dataset.py`, `run_process.py`,
and the modifications to the IQL files.

## Scope and status

Research code released to support the paper, not a maintained library. It is
the code used for the experiments, tidied for release: roughly 15 GB of
experiment logs and checkpoints removed, along with three unused modules that
had been copied in from sac-rnd but were never imported. No algorithmic changes
were made.

Reproducing the paper's numbers needs multiple seeds per D4RL task.

## Citation

```bibtex
@article{chaudhary2025from,
  title   = {From Novelty to Imitation: Self-Distilled Rewards for Offline Reinforcement Learning},
  author  = {Gaurav Chaudhary and Laxmidhar Behera},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2025},
  url     = {https://openreview.net/forum?id=F5K94JI2Jb}
}
```
