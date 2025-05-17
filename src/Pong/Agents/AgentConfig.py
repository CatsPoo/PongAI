from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AgenttConfig:
    epsilon:float
    epsilon_min:float
    epsilon_decay:float
    gamma:float
    lr:float

    #DQN Config
    buffer_size: int
    batch_size: int
    target_update: int
    


