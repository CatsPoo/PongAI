from Pong_env import PongEnv
from Agents_Model.DQN import DQN
import cv2

# def main():
#     env = PongEnv(render_height=200,render_width=400,render_mode='human')
#     obs = env.reset()

#     done = False
#     while (not done):
#         left_agent_action = env.action_space.sample()
#         right_agent_action = env.action_space.sample()

#         obs,left_reward,right_reward,done = env.step(left_agent_action,right_agent_action)
#         #randomly game loop

#         frame = env.render()
#         cv2.imshow('frame',frame)
#         cv2.waitKey(int(1000 / 30))

def train_agents(env:PongEnv,left_agent:DQN,right_agent:DQN,episodes):
    left_agent_rewards = []
    right_agent_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        left_agent_total_rewards = 0
        right_agent_total_rewards = 0

        while(not done):
            frame = env.render()
            if(frame.any()):
                cv2.imshow('frame',frame)
                cv2.waitKey(int(1000 /700))

            left_agent_action = left_agent.select_action(state)
            right_agent_action = right_agent.select_action(state)
            next_state, left_agent_reward,right_agent_reward, done = env.step(left_agent_action,right_agent_action)

            left_agent.buffer.push(state,left_agent_action,left_agent_reward,next_state,done)
            right_agent.buffer.push(state,right_agent_action,right_agent_reward,next_state,done)

            left_agent.train_step
            right_agent.train_step()

            state=next_state
            left_agent_total_rewards += left_agent_reward
            right_agent_total_rewards += right_agent_reward
        left_agent_rewards.append(left_agent_total_rewards)
        right_agent_rewards.append(right_agent_total_rewards)
        print(f"Episode {episode+1}/{episodes} | Total Reward (L/R): {left_agent_total_rewards:.2f}/{right_agent_total_rewards:.2f}| Epsilon (L/R): {left_agent.epsilon:.3f}/{right_agent.epsilon:.3f}")
    return left_agent_rewards, right_agent_rewards

def main():
    env = PongEnv(render_height=200,render_width=400,render_mode='human')
    env.reset()
    left_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    right_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    left_rewards,right_rewards = train_agents(env,left_agent,right_agent,10000)

if __name__ == '__main__':
    main()