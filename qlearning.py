import argparse
import sys
from typing import Any, Callable

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
from collections import deque
import random
import torch.nn as nn
import torch
import numpy as np


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class ApproxQValue(nn.Module):
    """
    Neural network that predicts the q-values for all actions for a given state.
    """
    def __init__(self, input_size, output_size, activation: Callable[[Any], Any] = nn.functional.relu):
        super(ApproxQValue, self).__init__()
        hidden_size = int(np.ceil((input_size + output_size) / 2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = activation
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


class NeuralAgent(object):

    def __init__(self, action_space, t, buffer):
        self.action_space = action_space
        self.t = t

    def act(self, observation, reward, done):
        obs = [float(i) for i in observation]
        Qaction = neural_network(torch.tensor(obs)).tolist()
        proba_action1 = np.exp(Qaction[0]/self.t)/sum(np.exp(np.array(Qaction)/self.t))
        rand = random.random()
        if rand < proba_action1:
            return 0
        else:
            return 1

    def learn(self, buffer):
        batch = sampling(buffer, 10)
        


def sampling(buffer, batch_size):
    return random.sample(list(buffer), batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    episode_count = 10
    reward = 0
    done = False
    reward_evolution = []
    buffer = deque(maxlen=100)
    agent = NeuralAgent(env.action_space, 0.1, buffer)

    neural_network = ApproxQValue(4, 2)

    for i in range(episode_count):
        interactions = 0
        sum_reward = 0
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            last_state = ob
            ob, reward, done, _ = env.step(action)
            sum_reward += reward
            # env.render()
            interactions += 1
            buffer.append({"state": last_state, "action": action, "next_state": ob, "reward": reward, "end_ep": done})
            if done:
                reward_evolution.append(sum_reward)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    plt.plot(reward_evolution)
    plt.title("Évolution des récompenses obtenues par l'agent au cours des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("Récompense")
    plt.show()
    print(buffer)
    print(sampling(buffer, 4))


