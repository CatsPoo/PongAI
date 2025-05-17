import numpy as np
import gymnasium as gym
import cv2

class PongEnv(gym.Env):

    def __init__(self,render_height = 200,render_width=400,peddal_length = 60,ball_size = 10, render_mode=None):

        self.env_ratio = 1/render_width
        self.height = render_height * self.env_ratio
        self.width = render_width * self.env_ratio
        self.peddal_length = peddal_length * self.env_ratio
        self.ball_size = ball_size * self.env_ratio

        self.render_mode = render_mode

        self.action_space  = gym.spaces.Discrete(3)      # up / stay / down

        env_shape = np.array([self.width,self.height])
        self.observation_space = gym.spaces.Dict({
            "ball_pos":   gym.spaces.Box(low = (-0.5 * env_shape) + self.ball_size,high=(0.5 * env_shape) - self.ball_size),
            "ball_vel":   gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32),
            "Left_Peddal_pos": gym.spaces.Box(low= -0.5 * self.height + self.peddal_length /2, high=0.5 * self.height - self.peddal_length /2, shape=(1,), dtype=np.float32),
            "Right_Peddal_pos": gym.spaces.Box(low= -0.5 * self.height + self.peddal_length/2, high=0.5 * self.height - self.peddal_length /2, shape=(1,), dtype=np.float32),
            "ball_size": gym.spaces.Box(low = 0,high=1,shape=(1,),dtype=np.float32)
        })

        self.left_peddat_center = None
        self.right_peddal_center = None
        self.ball_location = np.array([0.,0.])
        self.ball_vel = np.array([0.0055,0.01])

        self.reset()

    def _obs(self):
            return np.array([
        self.ball_location[0],
        self.ball_location[1],
        self.ball_vel[0],
        self.ball_vel[1],
        self.peddal_length,
        float(self.left_peddat_center),   # unwrap 1-element array → scalar
        float(self.right_peddal_center)], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        random_obs = self.observation_space.sample()
        self.left_peddat_center = random_obs['Left_Peddal_pos']
        self.right_peddal_center = random_obs['Right_Peddal_pos']
        self.ball_location = random_obs['ball_pos']

        return self._obs


    def step(self, left_agent_action,right_agent_action):
        # update paddle
        self.right_peddal_center = np.clip(self.right_peddal_center +  self.env_ratio * (right_agent_action - 1), -0.5 *self.height, 0.5 * self.height)
        self.left_peddat_center = np.clip(self.left_peddat_center +  self.env_ratio * (left_agent_action - 1), -0.5 *self.height, 0.5 * self.height)
        # update ball
        self.ball_location[0] += self.ball_vel[0]
        self.ball_location[1] += self.ball_vel[1]
        # reflect off top/bottom
        if abs(self.ball_location[1]) > self.height: 
            self.ball_vel[1] *= -1
        # check paddle bounce
        left_peddal_reward , right_peddal_reward , terminated = 0.,0., False

        #if ball in right wall
        if self.ball_location[0] + self.ball_size > self.width /2:
            if abs(self.ball_location[1] - (self.right_peddal_center)) < self.peddal_length:
                self.ball_vel[0] *= -1
                right_peddal_reward = 0.5
                left_peddal_reward = 0
            else:
                right_peddal_reward = -1.
                left_peddal_reward = 1
                terminated = True
        
        #if ball in left wall
        if self.ball_location[0] - self.ball_size < self.width /2:
            if abs(self.ball_location[1] - (self.left_peddat_center)) < self.peddal_length:
                self.ball_vel[0] *= -1
                left_peddal_reward = 0.5
                right_peddal_reward = 0
            else:
                right_peddal_reward = -1.
                left_peddal_reward = 1
                terminated = True

        return self._obs(), left_peddal_reward,right_peddal_reward, terminated

    def env2px(self,x,y = None):
        if (y):
            return int((x + self.width) / self.env_ratio),int((y+ self.height)/self.env_ratio)
        return int(x / self.env_ratio)
    
    def render(self):
        if self.render_mode != "human":
            return
        window_padding = 15
        paddle_thickness = 10
        rw ,rh = self.env2px(self.width,self.height)
        frame = np.full((rh, rw + (2 * paddle_thickness) + window_padding, 3),255,dtype=np.uint8)

        cv2.circle(frame, self.env2px(self.ball_location[0],self.ball_location[1]), self.env2px(self.ball_size),(0,0,0), thickness=-1)
        cv2.line(frame,(window_padding, self.env2px(0,self.left_peddat_center - self.peddal_length/2)[1]),(window_padding, self.env2px(0,self.left_peddat_center + self.peddal_length/2)[1]),(0,0,0),paddle_thickness)
        cv2.line(frame,(rw + paddle_thickness + window_padding, self.env2px(0,self.right_peddal_center - self.peddal_length/2)[1]),(rw+paddle_thickness + window_padding, self.env2px(0,self.right_peddal_center + self.peddal_length/2)[1]),(0,0,0),paddle_thickness)

        return frame