from Pong.Environment.Pong_env import PongEnv
from Pong.Agents.DQN import DQN
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def train_agents(env:PongEnv,left_agent:DQN,right_agent:DQN,episodes,fps=30):
    left_agent_rewards = []
    right_agent_rewards = []
    epsilons = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        left_agent_total_rewards = 0
        right_agent_total_rewards = 0

        while(not done):
            frame = env.render()
            if(frame.any()):
                cv2.imshow('frame',frame)
                cv2.waitKey(int(1000 /fps))

            left_agent_action = left_agent.select_action(state)
            right_agent_action = right_agent.select_action(state)
            next_state, left_agent_reward,right_agent_reward, done = env.step(left_agent_action,right_agent_action)

            left_agent.buffer.push(state,left_agent_action,left_agent_reward,next_state,done)
            right_agent.buffer.push(state,right_agent_action,right_agent_reward,next_state,done)

            left_agent.train_step()
            right_agent.train_step()

            state=next_state
            left_agent_total_rewards += left_agent_reward
            right_agent_total_rewards += right_agent_reward
        left_agent_rewards.append(left_agent_total_rewards)
        right_agent_rewards.append(right_agent_total_rewards)
        epsilons.append(left_agent.epsilon)

        #show_progress_plot(left_agent_rewards,right_agent_rewards,epsilons)
        print(f"Episode {episode+1}/{episodes} | Total Reward (L/R): {left_agent_total_rewards:.2f}/{right_agent_total_rewards:.2f}| Avg Reward (L/R): {np.mean(left_agent_rewards):.2f}/{np.mean(right_agent_rewards):.2f}| Epsilon (L/R): {left_agent.epsilon:.3f}/{right_agent.epsilon:.3f}")
    return left_agent_rewards, right_agent_rewards,epsilons


def show_progress_plot(left_rewards: list[float],right_rewards: list[float],
                       epsilons: list[float],
                       win_name = "Training Progress") -> None:

    # 1️⃣ Plot into an off-screen matplotlib figure
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Left Agent Reward", color="tab:blue")
    ax1.plot(left_rewards, color="tab:blue", label="Reward")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    fig, ax2 = plt.subplots(figsize=(6, 4), dpi=100)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Right Agent Reward", color="tab:blue")
    ax2.plot(right_rewards, color="tab:blue", label="Reward")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax3 = ax1.twinx()                              # 2nd y-axis
    ax3.set_ylabel("Epsilon", color="tab:red")
    ax3.plot(epsilons, color="tab:red", label="Epsilon")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout(pad=2.0)
    fig.canvas.draw()                              # render to RGBA buffer

    # # 2️⃣ Convert the figure to an RGB NumPy array
    # w, h = fig.canvas.get_width_height()
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # img = img.reshape((h, w, 3))
    # plt.close(fig)                                 # free the figure

    # # 3️⃣ OpenCV expects BGR
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # # 4️⃣ Display / update window
    # cv2.imshow(win_name, img_bgr)
    # cv2.waitKey(1)


def main():
    HERE = Path(__file__).resolve().parent
    env = PongEnv(render_mode='human')
    env.reset()

    left_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    right_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    # left_agent = DQN.load(HERE/'../Trained_Models/left')
    # right_agent = DQN.load(HERE/'../Trained_Models/right')

    left_rewards,right_rewards,epsilons = train_agents(env,left_agent,right_agent,100000,30)


    right_agent.save(HERE/'../Trained_Models/right')
    left_agent.save(HERE/'../Trained_Models/left')


if __name__ == '__main__':
    main()