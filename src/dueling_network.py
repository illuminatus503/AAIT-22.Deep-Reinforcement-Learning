import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNetwork(nn.Module):
    def __init__(
        self,
        lr: float,
        n_actions: int,
        name: str,
        input_dim: int,
        chkptr_dir: str,
        device: T.device | str = "cpu",
    ):
        super(DuelingDeepQNetwork, self).__init__()
        self.device = device
        self.chkptr_dir = chkptr_dir
        self.checkpoint_file = os.path.join(self.chkptr_dir, name)

        ## Arquitectura
        # Capa principal: detección inicial
        self.fc1 = nn.Linear(input_dim, 512, device=self.device)
        # Value function
        self.V = nn.Linear(512, 1, device=self.device)
        # Advantage function
        self.A = nn.Linear(512, n_actions, device=self.device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V_out = self.V(flat1)
        A_out = self.A(flat1)
        return V_out, A_out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
