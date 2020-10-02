'''
An implementation of the policy gradient algorithm (REINFORCE)
'''
import torch
import torch.nn as nn
import gym
from torch.optim.adam import Adam
from torch.optim import Adam
from torch.distributions.categorical import Categorical
import numpy as np


class CartPolePGNetwork(nn.Module):

    def __init__(self) -> None:
        super(CartPolePGNetwork, self).__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, state):
        return self.policy_net(state)


def train_pg(
    n_epochs=500,
    env=gym.make('CartPole-v0'),
    policy=CartPolePGNetwork(),
    reward_to_go=True,
):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        policy.cuda()
    optimizer = Adam(policy.parameters(), lr=1e-2)
    for i_epoch in range(n_epochs):
        batch_size = 5000
        states = []
        actions = []
        weights = []  # R(tau)
        rets = []
        lens = []
        state = env.reset()
        done = False
        ep_rews = []

        while True:
            states.append(state.copy())

            # evaluate current policy
            action = Categorical(logits=policy(torch.as_tensor(
                state, dtype=torch.float32, device=device))).sample().item()

            state, rew, done, _ = env.step(action)
            actions.append(action)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                rets.append(ep_ret)
                lens.append(ep_len)
                if reward_to_go:
                    rtgs = np.zeros_like(ep_rews)
                    # Reverse accumulated sum
                    for i in reversed(range(ep_len)):
                        rtgs[i] = ep_rews[i] + (rtgs[i+1] if i+1 < ep_len else 0)
                    weights += list(rtgs)
                else:
                    weights += [ep_ret] * ep_len
                state, done, ep_rews = env.reset(), False, []
                if len(states) > batch_size:
                    break
        optimizer.zero_grad()
        logp = Categorical(logits=policy(
            torch.as_tensor(states, dtype=torch.float32, device=device),
        )).log_prob(
            torch.as_tensor(actions, dtype=torch.int32, device=device),
        )
        loss = -(logp * torch.as_tensor(weights,
                                        dtype=torch.float32, device=device)).mean()
        loss.backward()
        optimizer.step()
        print(loss, np.mean(rets), np.mean(lens))


def main():
    train_pg()


if __name__ == '__main__':
    main()
