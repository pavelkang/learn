import os

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent
from param_def import DQNParam, TrainingParam


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


def gen_q_and_target_network(device, net_class):
    q_net = net_class()
    target_net = net_class()
    if device == torch.device('cuda:0'):
        q_net.cuda()
        target_net.cuda()
    target_net.eval()
    return q_net, target_net


def train_dqn(
    env,
    QNetClass,
    dqn_param: DQNParam,
    training_param: TrainingParam,
):
    q_net, target_net = gen_q_and_target_network(
        training_param.device, QNetClass)
    agent = DQNAgent(
        env,
        q_net,
        target_net,
        dqn_param,
        training_param,
        QNetClass.state_dim,
        QNetClass.action_dim,
    )
    global_t = 0
    eps = dqn_param.epsilon

    # Create directories if not exist
    log_path = 'logs/{}'.format(training_param.name)
    model_path = 'models/{}-models/'.format(training_param.name)
    for path in [log_path, model_path]:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    writer = SummaryWriter(log_path)
    for i_episode in range(training_param.n_episodes):
        obs = env.reset()
        done = False
        local_t = 0
        rewards_this_episode = []
        losses = []
        while not done:
            action = agent.choose_action(obs, eps)
            next_obs, reward, done, _ = env.step(action)
            rewards_this_episode.append(reward)
            agent.add_to_replay_buffer(obs, action, next_obs, reward, done)
            loss = agent.train(local_t, global_t)
            losses.append(loss)  # can be None
            eps = max(eps * eps, dqn_param.min_epsilon)
            global_t += 1
            local_t += 1
            obs = next_obs

        # logging
        if i_episode % training_param.log_every == 1:
            print("Episode", i_episode)
            total_reward = sum(rewards_this_episode)
            writer.add_scalar("total_reward_this_episode",
                              total_reward, i_episode)
            writer.add_scalar("avg_reward_this_episode",
                              total_reward/len(rewards_this_episode), i_episode)
            non_null_losses = list(filter(lambda x: x is not None, losses))
            writer.add_scalar("non_null_training_perc_this_episode", len(
                non_null_losses) / len(losses), i_episode)
            avg_loss = -1
            if len(non_null_losses) > 0:
                avg_loss = sum(
                    map(lambda x: x.item(), non_null_losses)) / len(non_null_losses)
            writer.add_scalar("loss_per_episode", avg_loss, i_episode)

            print('total reward:', total_reward)
            print('loss:', avg_loss)

        torch.save(agent.q_network.state_dict(), model_path+'model.pt')

    env.close()
    writer.flush()


def test(
    env,
    device,
    model_path='cartpole-models/1.pt',
    n_episodes=100,
):
    q_net = CartPoleNetwork()
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()
    agent = DQNAgent(
        env,
        q_net,
        None,
        1000000,
        0.99,
        32,
        100,
        device,
        10,
        1,
    )
    rewards = []
    for i_episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_this_episode = 0
        while not done:
            action = agent.choose_action(obs, -1)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            reward_this_episode += reward
        rewards.append(reward_this_episode)
    env.close()
    print('rewards:', rewards)
    return rewards


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env, device, dtype=torch.float32):
        super(ImageToPyTorch, self).__init__(env)
        self.dim = len(env.observation_space.shape)
        self.device = device
        self.dtype = dtype
        if self.dim >= 3:
            # Atari game images
            old_shape = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(
                old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
        elif self.dim == 1:
            # CartPole
            pass
        else:
            raise NotImplementedError

    def observation(self, observation):
        if self.dim >= 3:
            np_obs = np.swapaxes(observation, 2, 0)
            return torch.as_tensor(np_obs, dtype=self.dtype).to(self.device)
        elif self.dim == 1:
            return torch.as_tensor(observation, dtype=self.dtype).to(self.device)
        else:
            raise NotImplementedError


def train_cartpole_dqn():
    dqn_param = DQNParam(
        target_update_rate=200,
        double_dqn=True
    )
    training_param = TrainingParam(
        name='cartpole-dqn',
        n_episodes=3000
    )
    env = ImageToPyTorch(gym.make('CartPole-v0'), training_param.device)
    train_dqn(
        env,
        CartPoleNetwork,
        dqn_param,
        training_param,
    )

def train_mountain_car_dqn():
    dqn_param = DQNParam()
    training_param = TrainingParam(
        name='mountaincar-dqn',
    )
    env = ImageToPyTorch(gym.make('MountainCar-v0'), training_param.device)
    print(env.observation_space, env.action_space)
    train_dqn(
        env,
        MountainCarNetwork,
        dqn_param,
        training_param,
    )


def train_breakout_dqn():
    dqn_param = DQNParam(
        replay_buffer_size=1000
    )
    training_param = TrainingParam(
        name='breakout-dqn',
    )
    env = ImageToPyTorch(gym.make('Breakout-v0'), training_param.device,)
    train_dqn(
        env,
        BreakoutNetwork,
        dqn_param,
        training_param,
    )


def main():
    torch.cuda.empty_cache()
    # train_mountain_car_dqn()
    train_cartpole_dqn()
    # train(gym.make('CartPole-v0'), 3000, device)
    # test(gym.make('CartPole-v0'), torch.device('cpu'))


if __name__ == "__main__":
    main()
