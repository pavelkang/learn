# This module defines neural networks
# for processing a game's state information
# For atari games, we use CNN. For simpler scenarios, we
# use simple fully-connected linear layers.
import torch.nn as nn


class CartPoleNetwork(nn.Module):

    state_dim = (4,)
    action_dim = 2

    def __init__(self):
        super(CartPoleNetwork, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(self.state_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, features):
        return self.q_net(features)


class MountainCarNetwork(nn.Module):

    state_dim = (2,)
    action_dim = 3

    def __init__(self):
        super(MountainCarNetwork, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(self.state_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def forward(self, features):
        return self.q_net(features)


class BreakoutNetwork(nn.Module):

    state_dim = (3, 160, 210)
    action_dim = 4

    def __init__(self):
        super(BreakoutNetwork, self).__init__()
        self.q_net = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )

    def forward(self, features):
        x = self.q_net(features)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AtlantisNetwork(nn.Module):
    pass


class AtlantisDuelingNetwork(nn.Module):
    state_dim = (3, 160, 210)
    action_dim = 4

    def __init__(self) -> None:
        super(AtlantisDuelingNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        self.advantage_network = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.state_value_network = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        conv_features = self.conv_net(features)
        conv_features = conv_features.view(conv_features.size(0), -1)
        advantages = self.advantage_network(conv_features)
        state_value = self.state_value_network(conv_features)
        return state_value + advantages - advantages.mean(dim=-1, keepdim=True)
