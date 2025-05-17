from abc import ABC, abstractmethod

class AbstractAgent(ABC):
    def __init__(self,gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9999):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.step_count = 0
    
    @abstractmethod
    def select_action(self,state):
        pass

    def train_step(self):
        pass