import gymnasium as gym
import numpy as np
import seaborn as sns
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class CliffWalkingEnv(gym.Env):
    def __init__(self):
        super(CliffWalkingEnv, self).__init__()
        
        # 4x12のグリッド
        self.height = 4
        self.width = 12
        self.shape = (self.height, self.width)
        
        # アクション空間: 0: 上, 1: 右, 2: 下, 3: 左
        self.action_space = spaces.Discrete(4)
        
        # 状態空間: 0 から 47 までの離散的な状態
        self.observation_space = spaces.Discrete(self.height * self.width)
        
        # 初期状態と目標状態
        self.start_state = self.height * self.width - self.width
        self.goal_state = self.height * self.width - 1

        print("start_state:",self.start_state)
        print("goal_state:",self.goal_state)

        # 報酬の初期化
        self.rewards = np.zeros(self.shape)
        self.rewards.fill(-1)
        # デフォルトの報酬設定
        self.cliff_reward = -100
        self.rewards[3, 1:-1] = self.cliff_reward  # 崖（最下行の2列目から最後から2列目まで）
        self.rewards[2, 1:-1] = 5
        self.rewards[0, 1:-1] = 1

        self.goal_reward = 1
        self.rewards[self.goal_state // self.width, self.goal_state % self.width] = self.goal_reward  # ゴール
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return self.state, {}

    def step(self, action):
        i, j = self.state // self.width, self.state % self.width
        
        if action == 0:  # 上
            i = max(i - 1, 0)
        elif action == 1:  # 右
            j = min(j + 1, self.width - 1)
        elif action == 2:  # 下
            i = min(i + 1, self.height - 1)
        elif action == 3:  # 左
            j = max(j - 1, 0)
        
        new_state = i * self.width + j
        reward = self.rewards[i, j]

        done = (new_state == self.goal_state) or (reward == self.cliff_reward)

        if reward != -1 and not done:
            self.rewards[i, j] = -1  # 通過したセルの報酬を元に戻す

        self.state = new_state
        return self.state, reward, done, False, {}
    
    def set_reward(self):
        reward_0 = input("崖から一番離れている地点の報酬を設定してください:")
        # reward_1 = input("崖からちょっとだけ離れている地点の報酬を設定してください:")
        reward_2 = input("崖に一番近い地点の報酬を設定してください:")
        # reward_3 = input("崖から落ちてしまった時のの報酬を設定してください:")
        reward_4 = input("ゴール地点の報酬を設定してください:")


        self.rewards[0, 1:-1] = reward_0
        self.rewards[2, 1:-1] = reward_2
        self.goal_reward = reward_4


        self.rewards[self.goal_state // self.width, self.goal_state % self.width] = self.goal_reward
        self.rewards[self.start_state // self.width, self.start_state % self.width] = -1  # スタート地点

    def set_reward2(self, state, reward):
        i, j = state // self.width, state % self.width
        self.rewards[i, j] = reward

    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                state = i * self.width + j
                if state == self.state:
                    print("A", end=" ")  # エージェントの位置
                elif state == self.goal_state:
                    print("G", end=" ")  # ゴール
                elif self.rewards[i, j] == self.cliff_reward:
                    print("C", end=" ")  # 崖
                else:
                    print(".", end=" ")  # 通常のセル
            print()
        print()

    def plot_rewards(self):
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # 背景色を薄いグレーに設定
        ax.set_facecolor('#f0f0f0')

        # セルの色を設定
        colors = np.full(self.rewards.shape, '#ffffff')  # デフォルトは白
        colors[self.start_state // self.width, self.start_state % self.width] = '#87CEFA'  # スタート地点を薄い青に
        colors[self.goal_state // self.width, self.goal_state % self.width] = '#FFA07A'  # ゴール地点を薄い赤に

        # セルを描画
        for i in range(self.height):
            for j in range(self.width):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, facecolor=colors[i, j], edgecolor='gray'))
                ax.text(j, i, f'{self.rewards[i, j]:.1f}', ha='center', va='center', fontweight='bold')

        # 軸の設定
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))

        # グリッド線を追加
        # ax.grid(which="both", color="gray", linestyle='-', linewidth=1)

        # タイトルを設定
        plt.title("Rewards Map")
        
        # 凡例を追加
        start_patch = plt.Rectangle((0,0), 1, 1, fc='#87CEFA')
        goal_patch = plt.Rectangle((0,0), 1, 1, fc='#FFA07A')
        ax.legend([start_patch, goal_patch], ['Start', 'Goal'], loc='upper right', bbox_to_anchor=(1.1, 1))

        plt.tight_layout()
        plt.show()


class Inference:
    def __init__(self, q_table):
        data = np.load(q_table)
        self.q_table = data['q_table']

    def decide_action(self, state):
        action = np.argmax(self.q_table[state])
        return action
    
    def plot_q_table(self):
        # カスタムのセル値フォーマット関数を定義します
        def format_cell_value(x):
            coefficient = '{:.3g}'.format(x)
            return fr'${{{coefficient}}}$'


        # セルの値をX*10^nの形式で変換します
        formatted_data = np.vectorize(format_cell_value)(self.q_table / 100)

        # ヒートマップを作成します
        plt.figure(figsize=(10, 8))
        plt.title("Q-Table Visualization")
        plt.xlabel("Action")
        plt.ylabel("State")
        ax = sns.heatmap(self.q_table, annot=formatted_data, fmt="", cmap="YlGnBu",linewidths=.5)

        plt.show()