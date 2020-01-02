import argparse
import copy
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
import torch.nn.functional as F

from stopwatch import Stopwatch


MIN_EPOCH = 5000
LEARNING_RATE = 0.01
MOMENTUM = 0.9


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
    def __init__(self, input_size=4, output_size=2, activation: Callable[[Any], Any] = nn.functional.relu):
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

    def __init__(self, action_space, t, buffer, gamma):
        self.action_space = action_space
        self.t = t
        self.gamma = gamma
        self.nb_learn = 0

    def act(self, observation, reward, done):
        obs = [float(i) for i in observation]
        self.action_space = obs
        # print(torch.tensor(obs))
        Qaction = (neural_network(torch.tensor(obs)))
        # proba_action1 = np.exp(Qaction[0]/self.t)/sum(np.exp(np.array(Qaction)/self.t))
        # rand = random.random()
        # if rand < proba_action1:
        #     return 0
        # else:
        #     return 1
        _, action = torch.max(Qaction, 0)
        action = action.item()
        # print(action)
        # print(action)
        rand = random.random()
        if rand < self.t:
            rand = random.random()
            if rand < 0.5:
                return 0
            else:
                return 1
        else:
            return action

    def learn(self, buffer):
        global target_neural_network
        if len(buffer) > 100:
            batch = sampling(buffer, 42)
            self.nb_learn += 1
            for ex in batch:
                optim.zero_grad()
                qvalue = neural_network(torch.tensor(ex["state"], dtype=torch.float))
                if ex["end_ep"]:
                    j = (qvalue[ex["action"]].item() - (
                            ex["reward"])) ** 2
                else:
                    j = (qvalue[ex["action"]].item() - (
                            ex["reward"] + self.gamma * target_neural_network(torch.tensor(ex["next_state"], dtype=torch.float)).max().item())) ** 2
                # print(j)
                a = qvalue.clone()
                a[ex["action"]] = j
                #t = torch.tensor(j, requires_grad=True)
                #print(neural_network.weight.grad)
                loss = criterion(qvalue, a)
                loss.backward()
                # print(neural_network.weight.grad)
                optim.step()
            if self.nb_learn > 10:
                # print(neural_network.weight)
                # second_neural_network.load_state_dict(neural_network.state_dict())
                target_neural_network = copy.deepcopy(neural_network)
                self.nb_learn = 0
            self.t = max(self.t * 0.99, 0.001)
            # print(self.t)


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
    
    reward = 0
    done = False
    reward_evolution = []
    buffer = deque(maxlen=10000)
    agent = NeuralAgent(env.action_space, 1.0, buffer, 0.99)

    # neural_network = nn.Linear(4, 2)
    # second_neural_network = type(neural_network)(4, 2)
    # second_neural_network.load_state_dict(neural_network.state_dict())
    # print(list(neural_network.parameters()))
    neural_network = ApproxQValue(activation=F.relu)
    target_neural_network = copy.deepcopy(neural_network)
    print(list(neural_network.parameters()))
    criterion = nn.SmoothL1Loss()
    optim = torch.optim.SGD(neural_network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optim.zero_grad()
    learning_timer = Stopwatch()
    for i in range(MIN_EPOCH):
        interactions = 0
        sum_reward = 0
        ob = env.reset()

        while True:
            action = agent.act(ob, reward, done)
            # print(action)
            last_state = ob
            ob, reward, done, _ = env.step(action)
            sum_reward += reward
            # env.render()
            interactions += 1

            if done:
                buffer.append(
                    {"state": last_state, "action": action, "next_state": ob, "reward": reward, "end_ep": done})
                reward_evolution.append(sum_reward)
                break
            buffer.append({"state": last_state, "action": action, "next_state": ob, "reward": reward, "end_ep": done})

            # if interactions % 50 == 0:
            #    agent.learn(buffer)
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        
        if i % 100 == 0 and i != 0 or i == MIN_EPOCH:
            print("Train epoch {}/{} sum reward={}".format(i, MIN_EPOCH, sum_reward))
            print(agent.t)
        agent.learn(buffer)

    learning_timer.stop()
    
    # Close the env and write monitor result info to disk
    env.close()
    
    print("Learning time: {:.2f}s".format(learning_timer.elapsed()))
    plt.plot(reward_evolution)
    plt.title("Évolution des récompenses obtenues par l'agent au cours des itérations")
    plt.xlabel("Itérations")
    plt.ylabel("Récompense")
    plt.show()
    print(buffer)
    print(sampling(buffer, 4))


