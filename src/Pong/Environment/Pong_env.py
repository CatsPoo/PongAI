import numpy as np
import gymnasium as gym
import cv2
from Pong.Environment.EnviormentConfig import EnciormentConfig as ec
from Pong.Configurations.ConfigurationLoader import load_config
from pathlib import Path

class PongEnv(gym.Env):

    def __init__(self, render_mode=None):

        HERE = Path(__file__).resolve().parent
        self.env_cfg: ec = load_config(Path(HERE/'../../Config.yaml'),ec,'env')

        self.env_ratio = 1/self.env_cfg.width
        self.height = self.env_cfg.height * self.env_ratio
        self.width = self.env_cfg.width * self.env_ratio
        self.peddal_length = self.env_cfg.peddal_length * self.env_ratio
        self.ball_size = self.env_cfg.ball_size * self.env_ratio
        self.paddal_thickness = self.env_cfg.peddal_thickness * self.env_ratio
        self.render_mode = render_mode

        self.ball_catch_reward = self.env_cfg.ball_catch_reward
        self.ball_miss_reward= self.env_cfg.ball_miss_reward
        self.step_reward= self.env_cfg.step_reward
        self.goal_reward= self.env_cfg.goal_reward

        self.action_space  = gym.spaces.Discrete(3)      # up / stay / down

        env_shape = np.array([self.width,self.height])
        self.observation_space = gym.spaces.Dict({
            "ball_pos_x":   gym.spaces.Box(low = (-0.5 * env_shape[0]) + self.ball_size/2,high=(0.5 * env_shape[1]) - self.ball_size/2,shape=(1,), dtype=np.float32),
            "ball_pos_y":   gym.spaces.Box(low = (-0.5 * env_shape[1]) + self.ball_size/2,high=(0.5 * env_shape[1]) - self.ball_size/2,shape=(1,),dtype=np.float32),
            "ball_vel_y":   gym.spaces.Box(low=-self.env_cfg.ball_vel_y, high=self.env_cfg.ball_vel_y, shape=(1,), dtype=np.float32),
            "ball_vel_x":   gym.spaces.Box(low=-self.env_cfg.ball_vel_x, high=self.env_cfg.ball_vel_x, shape=(1,), dtype=np.float32),
            "Left_Peddal_pos": gym.spaces.Box(low= -0.5 * self.height + self.peddal_length /2, high=0.5 * self.height - self.peddal_length /2, shape=(1,), dtype=np.float32),
            "Right_Peddal_pos": gym.spaces.Box(low= -0.5 * self.height + self.peddal_length/2, high=0.5 * self.height - self.peddal_length /2, shape=(1,), dtype=np.float32),
        })

        self.left_peddat_center = None
        self.right_peddal_center = None
        self.ball_location = None
        self.ball_vel = None

        self.left_ball_touches =None
        self.right_ball_touches = None

        self.reset()

    def _obs(self):
        return np.array([
        self.ball_location[0],
        self.ball_location[1],
        self.ball_vel[0],
        self.ball_vel[1],
        float(self.left_peddat_center),   # unwrap 1-element array → scalar
        float(self.right_peddal_center)], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.left_ball_touches =0
        self.right_ball_touches=0
        self.ball_vel = np.array([self.env_cfg.ball_vel_x,self.env_cfg.ball_vel_y])
        if(np.random.rand()<0.5):self.ball_vel[0] *= -1
        if(np.random.rand()<0.5):self.ball_vel[1] *= -1
        random_obs = self.observation_space.sample()
        self.left_peddat_center = random_obs['Left_Peddal_pos']
        self.right_peddal_center = random_obs['Right_Peddal_pos']
        self.ball_location = np.array([random_obs['ball_pos_x'][0],random_obs['ball_pos_y'][0]]) 

        return self._obs()


    def get_observation_space_size(self):
        return len(self._obs())
    
    def step(self, left_agent_action,right_agent_action):
        # update paddle
        last_right_peddal_center = self.right_peddal_center
        last_left_peddal_center = self.left_peddat_center
        last_ball_location = self.ball_location

        self.right_peddal_center = np.clip(self.right_peddal_center +  self.env_ratio * self.env_cfg.peddal_speed * (right_agent_action - 1), -0.5 *self.height, 0.5 * self.height)
        self.left_peddat_center = np.clip(self.left_peddat_center +  self.env_ratio * self.env_cfg.peddal_speed * (left_agent_action - 1), -0.5 *self.height, 0.5 * self.height)
        # update ball
        self.ball_location[0] += self.ball_vel[0]
        self.ball_location[1] += self.ball_vel[1]

        # reflect off top/bottom
        if abs(self.ball_location[1]) > self.height: 
            self.ball_vel[1] *= -1
        # check paddle bounce
        left_peddal_reward , right_peddal_reward , terminated =  self.step_reward,self.step_reward, False

        last_left_peddat_ball_sistance = abs(last_right_peddal_center - last_ball_location[1])
        last_right_peddat_ball_sistance =abs(last_left_peddal_center - last_ball_location[1])
        current_left_pedat_ball_distance = abs(self.left_peddat_center - self.ball_location[1])
        current_right_pedat_ball_distance = abs(self.right_peddal_center - self.ball_location[1])

        if(current_left_pedat_ball_distance <= last_left_peddat_ball_sistance):
            left_peddal_reward  += self.env_cfg.distace_change_reward
        else:
            left_peddal_reward -= self.env_cfg.distace_change_reward

        if(current_right_pedat_ball_distance <= last_right_peddat_ball_sistance):
            right_peddal_reward  += self.env_cfg.distace_change_reward
        else:
            right_peddal_reward -= self.env_cfg.distace_change_reward

        #if ball in right wall
        if self.ball_location[0] - self.ball_size - self.paddal_thickness/4 > self.width:
            if abs(self.ball_location[1] - (self.right_peddal_center)) < self.peddal_length:
                self.ball_vel[0] *= -1
                right_peddal_reward += self.ball_catch_reward
                self.right_ball_touches+=1
            else:
                right_peddal_reward += self.ball_miss_reward
                left_peddal_reward += self.goal_reward * self.left_ball_touches
                terminated = True
        
        #if ball in left wall
        if self.ball_location[0]- self.ball_size - self.paddal_thickness /4< -self.width:
            if abs(self.ball_location[1] - (self.left_peddat_center)) < self.peddal_length:
                self.ball_vel[0] *= -1
                left_peddal_reward += self.ball_catch_reward
                self.left_ball_touches +=1
            else:
                left_peddal_reward += self.ball_miss_reward
                right_peddal_reward += self.goal_reward * self.right_ball_touches
                terminated = True

        return self._obs(), left_peddal_reward,right_peddal_reward, terminated

    def env2px(self,x,y = None):
        if (y):
            return int((x + self.width) / self.env_ratio),int((y+ self.height)/self.env_ratio)
        return int(x / self.env_ratio)
    
    def render(self):
        if self.render_mode != "human":
            return None
        
        window_padding = 15
        render_peddal_thickness = self.env2px(self.paddal_thickness)
        render_ball_size = self.env2px(self.ball_size)
        render_left_peddat_top = self.env2px(0,self.left_peddat_center - self.peddal_length/2)[1]
        render_left_peddat_bottom = self.env2px(0,self.left_peddat_center + self.peddal_length/2)[1]
        render_right_peddat_top = self.env2px(0,self.right_peddal_center - self.peddal_length/2)[1]
        render_right_peddal_bottom = self.env2px(0,self.right_peddal_center + self.peddal_length/2)[1]
        render_ball_location = self.env2px(self.ball_location[0],self.ball_location[1])

        rw ,rh = self.env2px(self.width,self.height)
        frame = np.full((rh, rw + (2 * render_peddal_thickness) + window_padding, 3),255,dtype=np.uint8)
        
        cv2.circle(frame, render_ball_location, render_ball_size,(0,0,0), thickness=-1)
        cv2.line(frame,(window_padding, render_left_peddat_top),(window_padding, render_left_peddat_bottom),(0,0,0),render_peddal_thickness)
        cv2.line(frame,(rw + render_peddal_thickness + window_padding, render_right_peddat_top),(rw+render_peddal_thickness + window_padding, render_right_peddal_bottom),(0,0,0),render_peddal_thickness)

        return frame