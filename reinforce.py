import numpy as np
import gym
import chainer
import chainer.functions as F
import chainer.links as L
import random
import math
from chainer import optimizers


class Policy(chainer.Chain):

    def __init__(self, n_mid_units=200, n_out=2):
        super(Policy, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            # self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        # h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h1))
        return F.softmax(h3)


class ReinforceAgent:

    def __init__(self, policy, optimizer, gamma=1.00):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma

    def act(self, obs):
        return self.policy(obs)

    def train(self, obs_sequences):

        loss = 0

        M = len(obs_sequences)

        for obs_seq in obs_sequences:

            reward_seq, action_probs_seq = obs_seq

            assert len(reward_seq) - 1 == len(action_probs_seq)

            T = len(action_probs_seq)

            # 時刻ごとの return (割引報酬和)を計算する
            discounteds = []
            for t in range(T):
                rewards_t = reward_seq[t + 1:]
                d = [r * (self.gamma ** i) for i, r in enumerate(rewards_t)]
                d = sum(d)
                discounteds.append(d)

            discounteds = chainer.Variable(np.array(discounteds, dtype=np.float32))
            action_probs = F.stack(action_probs_seq)

            loss += -F.sum(discounteds * F.log(F.max(action_probs, axis=1))) / (T * M)

        self.policy.cleargrads()
        loss.backward()
        self.optimizer.update()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    max_episode = 500
    num_episodes_to_train = 10

    policy = Policy()
    # optimizer = optimizers.Adam(alpha=1e-3)
    optimizer = optimizers.Adam(alpha=1e-3, adabound=True)
    optimizer.setup(policy)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0005))
    agent = ReinforceAgent(policy, optimizer)
    reward_history = []
    average_rewards = []

    episode = 0
    while episode < max_episode:

        obs_sequences = []
        for i in range(num_episodes_to_train):

            reward_seq = []
            action_probs_seq = []
            obs = env.reset()
            done = False
            total_reward = 0.0
            reward = 0
            frame = 0
            reward_seq.append(reward)
            while not done:
                obs = np.array([obs], dtype=np.float32)
                action_probs = agent.act(obs)[0]
                actions = np.argmax(action_probs.data)
                new_obs, _, done, info = env.step(actions)
                obs = np.array([new_obs], dtype=np.float32)
                frame += 1
                if done:
                    reward = 1 if frame >= 195 else -1
                else:
                    reward = 0.05
                action_probs_seq.append(action_probs)
                reward_seq.append(reward)
                total_reward += reward

                obs_sequences.append((reward_seq, action_probs_seq))

            reward_history.append(total_reward)

        agent.train(obs_sequences)
        episode += num_episodes_to_train

        average_reward = sum(reward_history[-20:]) / len(reward_history[-20:])
        print("average_reward = ", average_reward)
        average_rewards.append(average_reward)



    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.plot(average_rewards)
    plt.show()

    obs = env.reset()
    obs = np.array([obs], dtype=np.float32)
    done = False
    while not done:
        env.render()
        action_probs = agent.act(obs)[0]
        actions = np.argmax(action_probs.data)
        new_obs, _, done, info = env.step(actions)
        obs = np.array([new_obs], dtype=np.float32)
