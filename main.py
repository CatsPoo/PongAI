from Pong_env import PongEnv
import cv2

def main():
    env = PongEnv(render_height=200,render_width=400,render_mode='human')
    env.reset()
    frame = env.render()
    cv2.imshow('frame',frame)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()