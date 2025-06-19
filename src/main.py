from Pong.Environment.Pong_env import PongEnv
from Pong.Agents.DQN import DQN
from pathlib import Path
import numpy as np
from Pong.Display.Display import Display
from Pong.Trainer.Trainer import Trainer

HERE = Path(__file__).resolve().parent

def create_new_agents(env):
    left_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    right_agent = DQN(env.get_observation_space_size(),env.action_space.n)
    return left_agent,right_agent

def load_agent(train_time):
    left_agent = DQN.load(HERE/'../Trained_Models/left')
    right_agent = DQN.load(HERE/'../Trained_Models/right')
    return left_agent,right_agent

def main():
    episodes = 100000
    env = PongEnv(render_mode='human')
    env.reset()
    display = Display(200,400,50,True,600)

    left_agent,right_agent = create_new_agents(env)
    trainer = Trainer(env,left_agent,right_agent)
    left_rewards,right_rewards,epsilons = trainer.train(episodes,display)




    right_agent.save(HERE/'../Trained_Models/right')
    left_agent.save(HERE/'../Trained_Models/left')


if __name__ == '__main__':
    main()