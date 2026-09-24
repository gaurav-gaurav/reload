import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import utils
import copy
import pickle
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def mlp_critic(input_dimensions, output_dimensions, hidden_sizes=[400, 300], activation=nn.ReLU(), out_activation=None):
    layer_sizes = [input_dimensions] + hidden_sizes + [output_dimensions]
    modules = []
    for i in range(len(layer_sizes) - 2):  # Exclude the last layer for normalization/activation
        modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        modules.append(nn.LayerNorm(layer_sizes[i + 1]))  ## in orginal experiment this was applied
        modules.append(activation)

    # Add the final layer without LayerNorm
    modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    if out_activation is not None:
        modules.append(out_activation)

    return nn.Sequential(*modules)

class ConcatenationEncoder(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        observation_dim = observation_space.shape[0]
        self.output_dim = observation_dim
        
    def forward(self, observation, detach=False):
        observation = observation

        return observation

class ConvolutionalEncoder(nn.Module):
    def __init__(self, observation_space, feature_dim):
        """
        Assumes that the observation and goal images have the same dimensions.
        """
        super().__init__()
        
        assert observation_space['image_observation'].shape == (3, 84, 84)
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # compute output of flattened conv
        with torch.no_grad():
            n_flatten = self.conv(
                torch.as_tensor(observation_space['image_observation'].sample()).float().unsqueeze(0)
            ).shape[-1]
        
        self.head = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.LayerNorm(feature_dim))
        
        self.output_logits = False
        self.output_dim = 2 * feature_dim # double since outputs get concatenated
        
    def forward_single_observation(self, observation, detach=False):
        observation = observation / 255.
        
        observation = self.conv(observation)
        
        if detach:
            observation = observation.detach()
            
        observation = self.head(observation)
        
        return observation
        
    def forward(self, observation, detach=False):
        observation = self.forward_single_observation(observation, detach=detach)                
        return observation
    
    def copy_conv_weights_from(self, source):
        self.conv.load_state_dict(source.conv.state_dict())

class target_RND(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes):
        super().__init__()
        
        self.encoder = ConcatenationEncoder(observation_space)
        self.target_rnd = mlp_critic(2*self.encoder.output_dim,
                            32, [256, 256])

        self.apply(weight_init)  ### in orginal experiment this was applied
        for param in self.target_rnd.parameters():
            param.requires_grad =False

    def forward(self, observation, next_observation, detach_encoder=False):
        observation = self.encoder(observation, detach=detach_encoder)
        next_observation = self.encoder(next_observation, detach = detach_encoder)

        observation = torch.cat([observation, next_observation], dim=-1)
        with torch.no_grad():
            target_rnd = self.target_rnd(observation)
        return target_rnd

class predict_RND(nn.Module):
    def __init__(
            self,
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes):
        super().__init__()
        

        self.encoder = ConcatenationEncoder(observation_space)
        self.predict_rnd = mlp_critic(2*self.encoder.output_dim,
                            32, [256,256])
        #self.apply(weight_init)

    def forward(self, observation, next_observation, detach_encoder=False):
        observation = self.encoder(observation, detach=detach_encoder)
        next_observation = self.encoder(next_observation, detach = detach_encoder)

        observation = torch.cat([observation, next_observation], dim=-1)
        predict_rnd = self.predict_rnd(observation)
        return predict_rnd

class _RND(object):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(self,
            observation_space,
            action_space,
            feature_dim=32,
            hidden_sizes=[256, 256],
        ):
        self.device = torch.cuda.current_device()
        self.Target_RND = target_RND(
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes).to(
            self.device)

        self.Predict_RND = predict_RND(
            observation_space,
            action_space,
            feature_dim,
            hidden_sizes).to(
            self.device)

		