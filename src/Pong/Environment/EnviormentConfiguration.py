from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnciormentConfig:
    width: int
    height:int
    peddal_length: int
    ball_size: int

    ball_vel_x : float
    ball_vel_y: float

    peddal_thickness: int
    board_horizontal_padding:int


