from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnciormentConfig:
    width: int
    height:int
    peddal_length: int
    peddal_speed: float
    ball_size: int

    ball_vel_x : float
    ball_vel_y: float

    peddal_thickness: int
    board_horizontal_padding:int

    ball_catch_reward: float
    ball_miss_reward: float
    step_reward: float
    goal_reward: float
    distace_change_reward:float


