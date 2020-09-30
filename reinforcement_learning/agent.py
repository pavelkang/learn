from param_def import DQNParam, TrainingParam
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import torch

class DQNAgent:

    def __init__(
        self,
        env,
        q_network,
        target_network,
        dqn_param: DQNParam,
        training_param: TrainingParam,
        state_dim,
        action_dim,
    ):
        """
        env: OpenAI Gym env object
        """
        self.env = env
        self.q_network = q_network
        self.target_network = target_network
        if training_param.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                dqn_param.replay_buffer_size,
                training_param.batch_size,
                state_dim,
                action_dim,
                training_param.device,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                dqn_param.replay_buffer_size,
                training_param.batch_size,
                state_dim,
                action_dim,
                training_param.device,
            )
        self.gamma = dqn_param.gamma
        self.batch_size = training_param.batch_size
        self.target_update_rate = dqn_param.target_update_rate
        self.device = training_param.device
        self.learning_frequency = dqn_param.learning_frequency
        self.learning_start = dqn_param.learning_start
        self.optim = optim.Adam(self.q_network.parameters())
        self.double_dqn = dqn_param.double_dqn

    def _choose_random_action(self):
        return self.env.action_space.sample()

    def choose_action(
        self,
        obs: np.ndarray,
        eps,
    ):
        if random.random() < eps:
            return self._choose_random_action()
        q_values = self.q_network(obs.unsqueeze(0))
        return q_values.argmax(1).squeeze().item()

    def add_to_replay_buffer(
        self,
        state,
        action,
        next_state,
        reward,
        done
    ):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def train(self, local_t, global_t):
        if not self.replay_buffer.can_sample() or local_t < self.learning_start:
            return None
        sample = self.replay_buffer.sample(self.batch_size)
        state = sample['state']
        action = sample['action']
        reward = sample['reward']
        done = sample['done']
        next_state = sample['next_state']
        if self.double_dqn:
            next_state_max_action = self.q_network(next_state).argmax(dim=1, keepdim=True)
            next_state_max_qs = self.target_network(next_state).gather(-1, next_state_max_action).squeeze(1)
        else:
            next_state_max_qs = self.target_network(next_state).max(1)[0]
        qs = self.q_network(state).gather(-1, action.unsqueeze(-1))
        target_qs = reward + self.gamma * next_state_max_qs * (1 - done)
        loss = F.smooth_l1_loss(qs.squeeze(1), target_qs)
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            weights = torch.FloatTensor(sample['weights']).reshape(-1, 1).to(self.device)
            elementwise_loss = F.smooth_l1_loss(qs.squeeze(1), target_qs, reduction='none')
            loss = torch.mean(elementwise_loss * weights)
            self.replay_buffer.update_transition_priority(
                sample['idxs'],
                elementwise_loss.detach().cpu().numpy() + self.replay_buffer.prior_eps
            )

        self.optim.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        self._update_target_network(local_t, global_t)
        return loss

    def _update_target_network(self, local_t, global_t):
        if global_t % self.target_update_rate == self.target_update_rate-1:
            self.target_network.load_state_dict(self.q_network.state_dict())