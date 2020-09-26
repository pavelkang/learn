import random
from typing import List, Tuple
from segment_tree import SumSegmentTree
import numpy as np
import torch


class ReplayBuffer:

    '''
    A Pytorch-tensor based replay buffer
    '''

    def __init__(
        self,
        capacity: int,
        min_sample_size: int,
        state_shape: Tuple[int, int],
        action_dim: int,
        device: torch.device,
        dtype=torch.float32,
    ) -> None:
        '''
        capacity: max size of the buffer
        min_sample_size: minimum number of samples needed to be able to sample
        state_shape: Assumes state is of 2D shape (x, y)
        action_dim: Assumes action is 1D. assume discrete
        '''
        self.capacity = capacity
        self.min_sample_size = min_sample_size
        self.device = device
        self.state_buf = torch.zeros(
            (capacity, *state_shape), dtype=dtype, device=device)
        self.action_buf = torch.zeros(
            capacity, dtype=torch.long, device=device)
        self.next_state_buf = torch.zeros(
            (capacity, *state_shape), dtype=dtype, device=device)
        self.reward_buf = torch.zeros(capacity, dtype=dtype, device=device)
        self.done_buf = torch.zeros(capacity, dtype=torch.long, device=device)
        self.idx = 0
        self.size = 0

    def add(self, state, action: int, next_state, reward, done) -> None:

        self.state_buf[self.idx] = state
        self.action_buf[self.idx] = action
        self.next_state_buf[self.idx] = next_state
        self.reward_buf[self.idx] = reward
        self.done_buf[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def can_sample(self) -> bool:
        """
        Have enough transitions to be sampled
        """
        return self.size >= self.min_sample_size

    def sample(self, batch_size):
        idxs = torch.tensor(random.sample(
            range(self.size), batch_size)).to(self.device)
        return dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            next_state=self.next_state_buf[idxs],
            done=self.done_buf[idxs],
        )


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        capacity: int,
        min_sample_size: int,
        state_shape: Tuple[int, int],
        action_dim: int,
        device: torch.device,
        dtype = torch.float32,
        alpha: float = 0.6,
        prior_eps: float = 1e-6,
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            capacity, min_sample_size, state_shape, action_dim, device, dtype,
        )
        self.alpha = alpha
        self.sum_tree = SumSegmentTree(capacity)
        self.prior_eps = prior_eps
        self.max_priority = 1.0

    def add(self, state, action, next_state, reward, done) -> None:
        '''
        A newly added replay transition has max priority (1.0), and we store 1.0 ^ alpha
        to the sum segment tree
        '''
        self.sum_tree[self.idx] = self.max_priority ** self.alpha
        super(PrioritizedReplayBuffer, self).add(
            state, action, next_state, reward, done)

    def __len__(self):
        return self.size

    def _compute_importance_sampling_weight(self, idx, beta):
        Pi = self.sum_tree[idx] / self.sum_tree.sum()
        return (1 / (Pi * self.size)) ** beta

    def sample(self, batch_size, beta=0.4):
        '''
        '''
        # Sample indices with their probabilities
        total_sum = self.sum_tree.sum(0, self.size - 1)
        segment_length = total_sum / batch_size
        idxs = []
        weights = []
        max_weight = -float('inf')
        for i in range(0, batch_size):
            lower = i * segment_length
            upper = (i+1) * segment_length
            sample = random.uniform(lower, upper)
            idxs.append(self.sum_tree.retrieve(sample))
            weight = self._compute_importance_sampling_weight(idxs[-1], beta)
            max_weight = max(weight, max_weight)
            weights.append(weight)
        weights = list(map(lambda x: x/max_weight, weights))
        return dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            next_state=self.next_state_buf[idxs],
            done=self.done_buf[idxs],
            weights=weights,
            idxs=idxs,
        )

    def update_transition_priority(self, idxs, priorities):
        for i in range(len(priorities)):
            self.sum_tree[idxs[i]] = priorities[i]
            self.max_priority = max(self.max_priority, priorities[i])
