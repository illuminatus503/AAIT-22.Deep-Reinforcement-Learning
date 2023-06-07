import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNetwork(nn.Module):
    def __init__(
        self, lr: float, n_actions: int, name: str, input_dim: int, chkptr_dir: str
    ):
        super(DuelingDeepQNetwork, self).__init__()

        self.chkptr_dir = chkptr_dir
        self.checkpoint_file = os.path.join(self.chkptr_dir, name)

        self.fc1 = nn.Linear(input_dim, 512)
        self.V = nn.Linear(512, 1)  # Value function
        self.A = nn.Linear(512, n_actions)  # Action function

        self.optimiser = optim.Adam(self.parameters(), lr)
        self.loss = nn.MSELoss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))

        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
