import numpy as np
import gym
import chainer
import chainer.functions as F
import chainer.links as L
import random
from chainer import optimizers


class QFunction(chainer.Chain):

    def __init__(self, n_mid_units=32, n_out=2):
        super(QFunction, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# Experience replay 用
class ReplayBuffer:

    max_length = 1000
    minibatch_size = 32

    def __init__(self):
        self.buffer = []

    def push(self, replay):
        if len(self.buffer) >= self.max_length:
            self.buffer.pop()

        self.buffer.append(replay)

    def reset(self):
        self.buffer = []

    def get_minibatch(self):
        return random.sample(self.buffer, self.minibatch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(self, optimizer, gamma=0.99):
        self.q_func = QFunction()
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.optimizer = optimizer
        self.optimizer.setup(self.q_func)

    def get_action(self, observation, episode):
            # epsilon-greedy
            epsilon = np.random.rand()
            if epsilon < 1.0 / (1 + episode):
                rnd = random.randint(0, 1)
                return rnd

            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                out = self.q_func(observation).data[0]

            if out[0] >= out[1]:
                return 0
            else:
                return 1

    def train(self):
        if len(self.replay_buffer) < self.replay_buffer.minibatch_size:
            # replay buffer が溜まってないときは何もしない
            return

        # ミニバッチの準備
        trans = self.replay_buffer.get_minibatch()
        obs = np.array([t[0] for t in trans], dtype=np.float32)
        action = np.array([t[1] for t in trans], dtype=np.int32)
        new_obs = [t[2] for t in trans]
        reward = np.array([t[3] for t in trans], dtype=np.float32)

        # 教師信号を求める
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            q_max = np.zeros((self.replay_buffer.minibatch_size,),
                             dtype=np.float32)
            indices = np.zeros((len(new_obs,)), dtype=np.bool)
            values = []
            for i in range(len(new_obs)):
                if new_obs[i] is None:
                    indices[i] = False
                else:
                    indices[i] = True
                    values.append(new_obs[i])

            values = np.array(values, dtype=np.float32)
            q_max[indices] = F.max(self.q_func(values), axis=1).data

            q_supervisor = reward + self.gamma * q_max

        # 自分のQ値
        q = F.select_item(self.q_func(obs), action)

        # ロスを計算
        loss = F.mean_squared_error(q_supervisor, q)

        # 重みの更新
        self.q_func.cleargrads()
        loss.backward()
        self.optimizer.update()


class Trainer:

    def __init__(self, env, agent: DQNAgent):
        self.env = env
        self.agent = agent

    def run(self, max_episode):
        combo = 0  # 連続成功回数
        for episode in range(max_episode):
            obs = self.env.reset()
            total_reward = 0
            frame = 0
            # 1エピソードを試行
            while True:
                # 今回取る行動を決める
                obs_= np.array([obs], dtype=np.float32)
                action = self.agent.get_action(obs_, episode)
                # 行動を実行して、その結果の状態を観測する
                new_obs, _r, done, info = self.env.step(action)
                # 報酬はenvが返すものでなく、こちらで決める
                if done:
                    new_obs = None
                    if frame >= 195:
                        combo += 1
                        reward = +1
                    else:
                        combo = 0
                        reward = - 1
                else:
                    reward = 0.01
                total_reward += reward
                # リプレイを溜める
                replay = [obs, action, new_obs, reward]
                self.agent.replay_buffer.push(replay)
                # ネットワークの更新
                self.agent.train()
                # 観測値 = 今回の観測値 で更新
                obs = new_obs
                # ゲームオーバーの場合、試行を終わる
                frame += 1
                if done:
                    break
            print("episode {}: total_reward: {}".format(episode, total_reward))
            if combo >= 10:
                print("10回連続成功！")
                break


def main():
    env = gym.make('CartPole-v0')
    max_episode = 200
    optimizer = optimizers.Adam(alpha=0.001)
    agent = DQNAgent(optimizer, gamma=0.99)
    trainer = Trainer(env, agent)
    trainer.run(max_episode)

    # 最後に結果をレンダリング
    obs = env.reset()
    for i in range(200):
        obs = np.array([obs], dtype=np.float32)
        action = agent.get_action(obs, 1e10)
        obs, _, done, _, = env.step(action)
        env.render()
        if done:
            break

    env.close()


if __name__ == '__main__':
    main()