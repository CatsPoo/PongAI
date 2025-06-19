from Pong.Agents.AbstractAgent import AbstractAgent
from Pong.Environment.Pong_env import PongEnv
from Pong.Display.Display import Display

class Trainer:
    def __init__(self,env:PongEnv,left_agent:AbstractAgent,right_agent:AbstractAgent,):
        self.env = env
        self.loft_agent = left_agent
        self.right_agent = right_agent

    def train(self,episodes,display:Display = None):
        left_agent_rewards = []
        right_agent_rewards = []
        epsilons = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            left_agent_total_rewards = 0
            right_agent_total_rewards = 0
            while(not done):
                game = self.env.render()
                if(display):
                    display.update(game,left_agent_rewards,right_agent_rewards,epsilons)

                left_agent_action = self.left_agent.select_action(state)
                right_agent_action = self.right_agent.select_action(state)
                next_state, left_agent_reward,right_agent_reward, done = self.env.step(left_agent_action,right_agent_action)

                self.left_agent.buffer.push(state,left_agent_action,left_agent_reward,next_state,done)
                self.right_agent.buffer.push(state,right_agent_action,right_agent_reward,next_state,done)

                self.left_agent.train_step()
                self.right_agent.train_step()

                state=next_state
                left_agent_total_rewards += left_agent_reward
                right_agent_total_rewards += right_agent_reward
            left_agent_rewards.append(left_agent_total_rewards)
            right_agent_rewards.append(right_agent_total_rewards)
            epsilons.append(self.left_agent.epsilon)

            #show_progress_plot(left_agent_rewards,right_agent_rewards,epsilons)
            print(f"Episode {episode+1}/{episodes} | Total Reward (L/R): {left_agent_total_rewards:.2f}/{right_agent_total_rewards:.2f}| Avg Reward (L/R): {np.mean(left_agent_rewards):.2f}/{np.mean(right_agent_rewards):.2f}| Epsilon (L/R): {left_agent.epsilon:.3f}/{right_agent.epsilon:.3f}")
        return left_agent_rewards, right_agent_rewards,epsilons   
