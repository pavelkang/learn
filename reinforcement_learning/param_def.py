import torch
from dataclasses import dataclass


@dataclass
class DQNParam:
    replay_buffer_size: int = 10000
    gamma: float = 0.99
    target_update_rate: int = 100
    learning_frequency: int = 10
    learning_start: int = 1
    epsilon: float = 0.99
    min_epsilon: float = 0.1
    double_dqn: bool = False


@dataclass
class TrainingParam:
    name: str
    n_episodes: int = 1000
    batch_size: int = 32
    device: torch.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    log_every: int = 10
