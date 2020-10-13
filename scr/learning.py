#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import csv
import torch
import random
import math
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import OP3Env

H = 50 # 隠れ層のユニット数
N = 4 # 数値表記時の有効数字
CAPACITY = 1000000
NUM_EPISODES = 5000
MAX_STEP = 10000000
BATCH_SIZE = 32
GAMMA = 0.99
FILE_NAME = 'log/learning_log.csv'
RECORD_EPI = 1 # csvに保存する際のエピソード間隔

#namedtupleを作成
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY #メモリの長さ
        self.memory = [] #経験を保存する変数
        self.index = 0 #保存する番号を表す

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None) #メモリを追加
            #メモリにデータを保存
            self.memory[self.index] = Transition(state, action, state_next, reward)
            #indexをずらす
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        #batchサイズ分だけランダムに保存内容を取り出す
        return random.sample(self.memory, batch_size)

    def __len__(self):
        #関数lenに対して現在のmemoryの長さを返す
        return len(self.memory)


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        #ニューラルネットワークを構成
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, H))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(H,H))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(H, num_actions))
        print(self.model)
        #重みを学習する際の最適化手法の選択と設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        #メモリサイズの確認
        if len(self.memory) < BATCH_SIZE:
            return
        # memoryからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)
        #形式変換
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        #教師信号となるQ(s_t,a_t)を求める
        #ネットワークを推論モードに切り替える
        self.model.eval()
        #ネットワークが出力したQを求める
        state_action_values = self.model(state_batch).gather(1, action_batch)
        # max Q(s_t+1, a_t)値を求める
        # next_stateがあるかどうかをチェック
        non_final_mask = torch.ByteTensor(tuple(map(lambda s:s is not None, batch.next_state)))
        #とりま初期化
        next_state_values = torch.zeros(BATCH_SIZE)
        #next_stateを求める
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        #教師信号となるQ(s_t,a_t)をQ学習の式から求める
        expected_state_action_values = reward_batch + GAMMA*next_state_values

        #ネットワークを訓練モードに
        self.model.train()
        #損失関数を計算する
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        Environment.loss = loss

        #結合パラメータを更新する
        self.optimizer.zero_grad() #勾配をリセット
        loss.backward() #バックプロパゲーション
        self.optimizer.step() #結合パラメータの更新

    def decide_action(self, state, episode):
        epsilon = 0.5*(0.99**episode)

        if epsilon > np.random.uniform(0,1):
            #ネットワークを推論モードに
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1) #ネットワーク出力の最大値をだす
            # 0,1の行動をランダムに返す
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        # print(action)
        return action

# エージェント
class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        # self.state = None
        # self.action = 0
        # self.start_time = 0
        # self.trial = 0
        # self.episode = 0
        # self.reward = 0
        # self.dis = 0
        # self.data = []
        # self.last_index = 0
        # self.history = []

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

#実行環境設定
class Environment:
    def __init__(self):
        # 環境定義
        self.env = OP3Env.OP3Env()
        # statesやactionsの次元設定
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        Environment.loss = 0
        # 環境内で行動するAgentを生成
        self.agent = Agent(self.num_states, self.num_actions)

    #実行処理
    def run(self):
        # frames = [] # 最後の試行を動画にするために画像を格納する変数
        #各エピソードに行う動作
        for episode in range(NUM_EPISODES):
            #環境の初期化
            step = 0
            reward_sum = 0
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            # FloatTensor size 4をsize 1×4に変換
            state = torch.unsqueeze(state, 0)

            #各ステップに行う動作
            for step in range(MAX_STEP):
                action = self.agent.get_action(state,episode)
                observation_next, _, done, _ = self.env.step(action.item())
                # 報酬
                # reward, reward_num = self.get_reward(observation_next)
                reward_num = (observation_next[0] - state[0][0])*10
                reward_sum += reward_num/10
                reward = torch.FloatTensor([reward_num])
                #ファイル書き込み・通知
                if done:
                    state_next = None
                    dis_x = math.floor(observation_next[0] * 10 ** N)/(10 ** N)
                    dis_y = math.floor(observation_next[1] * 10 ** N)/(10 ** N)
                    rew = math.floor(reward_sum * 10 ** N)/(10 ** N)
                    print("\n=============episode {0}    終了====================\n"\
                          "reward: {1},  dis_x:{2},  dis_y:{3}\n"\
                          "==================================================\n".format(episode+1, rew, dis_x, dis_y))
                    if episode == 0:
                        df = pd.DataFrame([["episode", "reward_sum", "X_direction_distance", "Y_direction_distance", "Loss"]])
                        df.to_csv(FILE_NAME, index=False, encoding="utf-8")
                    if (episode % RECORD_EPI) == 0:
                        df = pd.DataFrame([[episode+1, reward_sum, observation_next[0], observation_next[1], Environment.loss]])
                        df.to_csv(FILE_NAME, index=False, encoding="utf-8", mode='a', header=False)
                        writelist = np.array([[episode+1, reward_sum, observation_next[0], observation_next[1], Environment.loss]], dtype=str)
                        print("以下を書き込みました\n", writelist, "\n")
                    break

                else:
                    # reward関係が本来なら入る
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    # FloatTensor size 4をsize 1×4に変換
                    state_next = torch.unsqueeze(state_next, 0)
                    # memoryに経験を追加
                    self.agent.memorize(state, action, state_next, reward)
                    # Q関数を更新
                    self.agent.update_q_function()
                    # 観測の更新
                    state = state_next

            #保存
            if (episode % 50 == 0):
                save_name = "saved_model/model" + str(episode) + ".pth"
                torch.save(self.agent.brain.model.state_dict(), save_name)



    # def get_reward(self, state):
    #     #進んだ距離を取得
    #     #dis_x = state[0]
    #     #横にずれた幅
    #     #dis_y = abs(state[1])
    #     #報酬計算
    #     #dis = self.calc_dis(dis_x, dis_y)
    #     #reward_num = (dis - dis_y)/10
    #     #if dis_x < 0:
    #     #    reward_num += dis_x/10
    #     reward_num = state[0]/10
    #     reward = torch.FloatTensor([reward_num])
    #     return reward, reward_num

    def calc_dis(self, dis_x, dis_y):
        dis = np.sqrt(dis_x**2 + dis_y**2)
        return dis


def main():
    OP3 = Environment()
    OP3.run()


if __name__== "__main__":
    main()
