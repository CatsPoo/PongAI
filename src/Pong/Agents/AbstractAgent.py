from abc import ABC, abstractmethod
from pathlib import Path
from Pong.Configurations.ConfigurationLoader import load_config
from Pong.Agents.AgentConfig import AgenttConfig as ac
import cloudpickle as cp
class AbstractAgent(ABC):
    def __init__(self):
        HERE = Path(__file__).resolve().parent
        self.agent_cfg: ac = load_config(Path(HERE/'../../Config.yaml'),ac,'agent')

        self.gamma = self.agent_cfg.gamma
        self.lr = self.agent_cfg.lr
        self.epsilon = self.agent_cfg.epsilon
        self.epsilon_min = self.agent_cfg.epsilon_min
        self.epsilon_decay = self.agent_cfg.epsilon_decay

        self.step_count = 0
    
    @abstractmethod
    def select_action(self,state):
        pass

    def train_step(self):
        pass
    
    def save(self, file: str | Path) -> None:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("wb") as f:
            cp.dump(self, f)

    @classmethod
    def load(cls, file: str | Path):
        with Path(file).open("rb") as f:
            return cp.load(f)