from Pong_env import PongEnv
import cv2

def main():
    env = PongEnv(render_height=200,render_width=400,render_mode='human')
    obs = env.reset()

    done = False
    while (not done):
        left_agent_action = env.action_space.sample()
        right_agent_action = env.action_space.sample()

        obs = env.step(left_agent_action,right_agent_action)
        #randomly game loop

        frame = env.render()
        cv2.imshow('frame',frame)
        cv2.waitKey(int(1000 / 30))

if __name__ == '__main__':
    main()