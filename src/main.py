from Pong.Environment.Pong_env import PongEnv
from Pong.Agents.DQN import DQN
from pathlib import Path
import numpy as np
from Pong.Display.Display import Display

def train_agents(env:PongEnv,left_agent:DQN,right_agent:DQN,episodes,display:Display):
    left_agent_rewards = []
    right_agent_rewards = []
    epsilons = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        left_agent_total_rewards = 0
        right_agent_total_rewards = 0
        while(not done):
            game = env.render()
            if(game.any()):
                display.update(game,left_agent_rewards,right_agent_rewards,epsilons)

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

def main():
    HERE = Path(__file__).resolve().parent
    env = PongEnv(render_mode='human')
    env.reset()

    left_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    right_agent = DQN(env.get_observation_space_size(),env.action_space.n)

    display = Display(200,400,50,True,24)
    # left_agent = DQN.load(HERE/'../Trained_Models/left')
    # right_agent = DQN.load(HERE/'../Trained_Models/right')

    left_rewards,right_rewards,epsilons = train_agents(env,left_agent,right_agent,100000,display)




    right_agent.save(HERE/'../Trained_Models/right')
    left_agent.save(HERE/'../Trained_Models/left')


if __name__ == '__main__':
    main()