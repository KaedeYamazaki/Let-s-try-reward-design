from Algorithms import Sarsa, Q_Learning
from OC_Cliff_Walking import CliffWalkingEnv,Inference
import gymnasium as gym
from tqdm import tqdm
import socket
import time


NUM_DIZITIZED = 6
ALPHA = 0.5
GAMMA = 0.99
EPSILON = 0.05
gym.envs.registration.register(id='KitOcEnv-v0',entry_point=CliffWalkingEnv)


episode_num = []
episode_rewards = []
episode_steps = []
env = gym.make('KitOcEnv-v0')


HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))


def Learning_Sarsa():
    
    print("Action_num:",env.action_space.n)
    print("Obs_num:",env.observation_space.n)

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    alpha = ALPHA
    gamma = GAMMA
    epsilon = EPSILON

    Sarsa_agent = Sarsa(num_actions, num_states, alpha, gamma, epsilon)

    print("Learning start")

    for i_episode in tqdm(range(5000)):
        state = env.reset()
        state = state[0]
        action = Sarsa_agent.decide_action(state)
        episode_reward = 0

        for t in range(1000):
            next_state, reward, done, _, _ = env.step(action)
            next_action = Sarsa_agent.decide_action(next_state)
            Sarsa_agent.update_Q(state, action, reward, next_state, next_action)
            action = next_action

            episode_reward += reward
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break

        episode_num.append(i_episode)
        episode_rewards.append(episode_reward)
        episode_steps.append(t)

    Sarsa_agent.save_Qtabele("KitOcEnv_v0")
    env.close()
    print("Learning finish")

def Learning_Qlearning():

    print("Action_num:",env.action_space.n)
    print("Obs_num:",env.observation_space.n)

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    alpha = ALPHA
    gamma = GAMMA
    epsilon = EPSILON

    Q_agent = Q_Learning(num_actions, num_states, alpha, gamma, epsilon)

    print("Learning start")

    for i_episode in tqdm(range(500)):
        state = env.reset()
        state = state[0]
        episode_reward = 0

        for t in range(100):
            action = Q_agent.decide_action(state)
            next_state, reward, done, _, _ = env.step(action)
            Q_agent.update_Q(state, action, next_state, reward)
            state = next_state

            episode_reward += reward
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break

        episode_num.append(i_episode)
        episode_rewards.append(episode_reward)
        episode_steps.append(t)

    Q_agent.save_Qtabele("KitOcEnv_v0")
    # env.close()
    print("Learning finish")

def Test():

    Q_agent = Inference("KitOcEnv_v0_Q_Learning_fail.npz")
    # Q_agent.plot_q_table()
    print("Inference start")

    state = env.reset()
    state = state[0]
    action = Q_agent.decide_action(state)
    episode_reward = 0
    print("")
    env.render()

    try:
        for t in range(36):
            next_state, reward, done, game_over, actual_actions = env.step(action)
            result = str(actual_actions['action'])
            print(result)
            client.sendall(result.encode('utf-8'))
            time.sleep(1.0)
            print("step:",t+1)
            env.render()
            next_action = Q_agent.decide_action(next_state)
            action = next_action

            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Your Score:",(((36-t)/36)*1000))
                break

            if t == 35 or game_over:
                print("GAME OVER")
                break

        env.close()
        print("Inference finish")

    finally:
        client.close()

def plot():
    import matplotlib.pyplot as plt
    plt.plot(episode_num, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episode')
    plt.show()

if __name__ == '__main__':
    print("OC_Cliff_Walking")
    print("崖に落ちないようにしながら最短経路でゴールを目指そう")

    env.set_reward()
    env.plot_rewards()
    # Learning_Sarsa()
    # Learning_Qlearning()
    plot()
    Test()