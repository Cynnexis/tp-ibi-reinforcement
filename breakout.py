import argparse
import copy
import sys
from collections import deque
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

import gym
from gym import wrappers, logger

from stopwatch import Stopwatch

MIN_EPOCH = 5000
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.001


class ConvNet(nn.Module):
    """
    Neural network that predicts the q-values for all actions for a given state.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.fc1 = nn.Linear(2592 * 2, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        print(self.action_space.sample())
        return self.action_space.sample()


class ConvAgent(object):
    def __init__(self, action_space, t, buffer, gamma):
        self.action_space = action_space
        self.t = t
        self.gamma = gamma
        self.nb_learn = 0

    def act(self, observation, reward, done):
        self.action_space = observation
        # print(torch.tensor(obs))
        # proba_action1 = np.exp(Qaction[0]/self.t)/sum(np.exp(np.array(Qaction)/self.t))
        # rand = random.random()
        # if rand < proba_action1:
        #     return 0
        # else:
        #     return 1
        # print(action)
        # print(action)
        rand = random.random()
        if rand < self.t:
            rand = random.random()
            if rand < 0.25:
                return 0
            elif rand < 0.5:
                return 1
            elif rand < 0.75:
                return 2
            else:
                return 3
        else:
            tens = torch.tensor(observation, dtype=torch.float)/255
            Qaction = (neural_network(tens))
            _, action = torch.max(Qaction, 0)
            action = action.item()
            return action

    def learn(self, buffer):
        global target_neural_network
        if len(buffer) > 5000:
            batch = sampling(buffer, 42)
            self.nb_learn += 1
            for ex in batch:
                optim.zero_grad()
                tens1 = torch.tensor(ex["state"], dtype=torch.float)/255
                qvalue = neural_network(tens1)
                if ex["end_ep"]:
                    j = (qvalue[ex["action"]].item() - (
                            ex["reward"])) ** 2
                else:
                    tens2 = torch.tensor(ex["next_state"], dtype=torch.float)/255
                    j = (qvalue[ex["action"]].item() - (
                            ex["reward"] + self.gamma * target_neural_network(tens2).max().item())) ** 2
                # print(j)
                a = qvalue.clone()
                a[ex["action"]] = j
                #t = torch.tensor(j, requires_grad=True)
                #print(neural_network.weight.grad)
                loss = criterion(qvalue, a)
                loss.backward()
                # print(neural_network.weight.grad)
                optim.step()
                # target_neural_network.fc1.weight = target_neural_network.fc1.weight * (1 - ALPHA) + ALPHA * neural_network.fc1.weight
            if self.nb_learn > 30:
                # print(neural_network.weight)
                # second_neural_network.load_state_dict(neural_network.state_dict())
                target_neural_network = copy.deepcopy(neural_network)
                self.nb_learn = 0
            self.t = max(self.t * 0.99, 0.001)
            print(self.t)


def sampling(buffer, batch_size):
    return random.sample(list(buffer), batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Breakout-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    env.spec.id += " NoFrameskip"

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env = wrappers.AtariPreprocessing(env, screen_size=84, frame_skip=4, grayscale_obs=True)
    env = wrappers.FrameStack(env, 4)
    env.seed(0)

    neural_network = ConvNet()
    target_neural_network = copy.deepcopy(neural_network)
    print(list(neural_network.parameters()))
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(neural_network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optim.zero_grad()
    reward = 0
    buffer = deque(maxlen=10000)
    done = False
    reward_evolution = []
    sum_reward = 0
    interactions = 0
    agent = ConvAgent(env.action_space, 1, buffer, 0.9)

    learning_timer = Stopwatch()
    # agent = RandomAgent(env.action_space)

    for i in range(MIN_EPOCH):
        sum_reward = 0
        interactions = 0
        print(i)
        ob = env.reset()
        ob = [[k for k in ob]]
        while True:
            action = agent.act(ob, reward, done)
            last_state = ob
            ob, reward, done, _ = env.step(action)
            ob = [[k for k in ob]]
            sum_reward += reward
            # env.render()
            interactions += 1
            if reward > 0:
                reward = 1
            if reward < 0:
                reward = -1

            if done:
                buffer.append(
                    {"state": last_state, "action": action, "next_state": ob, "reward": reward, "end_ep": done})
                reward_evolution.append(sum_reward)
                break
            buffer.append({"state": last_state, "action": action, "next_state": ob, "reward": reward, "end_ep": done})
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