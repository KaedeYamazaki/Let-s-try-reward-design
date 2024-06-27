import numpy as np
import random

class Q_Learning:
    def __init__(self, num_actions, num_states,alpha,gamma,epsilon):
        '''args
        num_actions: int, 行動の数
        num_states: int, 状態の数
        num_dizitized: int, 離散化の数
        alpha: float, 学習率
        gamma: float, 割引率
        epsilon: float, ε-greedy法のε
        '''
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.random.uniform(low=-1, high=1, size=(self.num_states, self.num_actions))

    def update_Q(self, state, action, next_state, reward):
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

    def decide_action(self, state):

        if self.epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = random.randrange(self.num_actions)

        return action

    def save_Qtabele(self,name):
        q_table_np = np.array(self.q_table)
        name = name + "_Q_Learning.npz"
        np.savez_compressed(name, q_table=q_table_np) # gzip圧縮しながら保存


class Sarsa:
    def __init__(self, num_actions, num_states, alpha,gamma,epsilon):
        '''args
        num_actions: int, 行動の数
        num_states: int, 状態の数
        num_dizitized: int, 離散化の数
        alpha: float, 学習率
        gamma: float, 割引率
        epsilon: float, ε-greedy法のε
        '''
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.random.uniform(low=-1, high=1, size=(self.num_states, self.num_actions))

    def update_Q(self, state, action, reward, next_state, next_action):
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])

    def decide_action(self, state):
        if self.epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = random.randrange(self.num_actions)

        return action

    def save_Qtabele(self,name):
        q_table_np = np.array(self.q_table)
        name = name + "_Sarsa.npz"
        np.savez_compressed(name, q_table=q_table_np) # gzip圧縮しながら保存


class Expected_Sarsa:
    def __init__(self, num_actions, num_states, alpha,gamma,epsilon):
        '''args
        num_actions: int, 行動の数
        num_states: int, 状態の数
        num_dizitized: int, 離散化の数
        alpha: float, 学習率
        gamma: float, 割引率
        epsilon: float, ε-greedy法のε
        '''
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.random.uniform(low=-1, high=1, size=(self.num_states, self.num_actions))

    def update_Q(self, state, action, reward, next_state):
        # 各行動を選択する確率
        policy_probs = np.ones_like(self.q_table[next_state]) * self.epsilon / self.num_actions
        policy_probs[np.argmax(self.q_table[next_state])] += (1 - self.epsilon)

        # 期待値を計算してQ値を更新
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * np.sum(policy_probs * self.q_table[next_state]) - self.q_table[state][action])

    def decide_action(self, state):
        if self.epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = random.randrange(self.num_actions)

        return action

    def save_Qtabele(self,name):
        q_table_np = np.array(self.q_table)
        name = name + "_Expected_Sarsa.npz"
        np.savez_compressed(name, q_table=q_table_np) # gzip圧縮しながら保存