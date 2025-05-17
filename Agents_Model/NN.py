import torch.nn as nn

# --- Q-Network ---
class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
