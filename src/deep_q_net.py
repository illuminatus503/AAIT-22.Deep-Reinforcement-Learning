import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models


class AbstractDQN(nn.Module):
    def __init__(
        self,
        lr: float,
        n_actions: int,
        input_dim: int,
        chkpoint_file: str,
        device: T.device | str,
    ):
        super(AbstractDQN, self).__init__()
        self._device = device
        self._checkpoint_file = chkpoint_file

        # ARQUITECTURA!!

        self.to(self._device)

    def forward(self, state):
        return T.zeros(1, device=self._device)

    def get_action(self, state):
        actions = self.forward(state)
        return T.argmax(actions).item()

    def save_checkpoint(self):
        T.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self._checkpoint_file))


class BasicDeepQNetwork(AbstractDQN):
    def __init__(
        self,
        lr: float,
        n_actions: int,
        input_dim: int,
        chkpoint_file: str,
        device: T.device | str,
    ):
        super(BasicDeepQNetwork, self).__init__(
            lr, n_actions, input_dim, chkpoint_file, device
        )

        ## Arquitectura
        # Capa principal: detección inicial
        self._fc1 = nn.Linear(input_dim, 512, device=self._device)
        self._fc2 = nn.Linear(512, 512, device=self._device)
        # Q_function
        self._Q = nn.Linear(512, n_actions, device=self._device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self._device)

    def forward(self, state):
        flat1 = F.relu(self._fc1(state))
        flat2 = F.relu(self._fc2(flat1))
        actions = self._Q(flat2)
        return actions


class DuelingDeepQNetwork(AbstractDQN):
    @staticmethod
    def __flatten_features(in_size):
        """
        output_size = (input_size + 2 * padding - kernel_size) / stride + 1
        """
        out1_s = (in_size + 2 * 0 - 8) / 1 + 1
        out2_s = (out1_s + 2 * 0 - 5) / 1 + 1
        out3_s = (out2_s + 2 * 0 - 3) / 1 + 1
        return out3_s * out3_s * 64

    def __init__(
        self,
        lr: float,
        n_actions: int,
        input_dim: int,
        chkpoint_file: str,
        device: T.device | str,
    ):
        super(DuelingDeepQNetwork, self).__init__(
            lr, n_actions, input_dim, chkpoint_file, device
        )

        flatten_features = 83 * 83 * 64  # Para una entrada de 96x96
        self._seq = nn.Sequential(
            nn.Conv2d(3, 32, 8, device=self._device),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, device=self._device),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, device=self._device),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flatten_features, 64, device=self._device),
        )

        # Value function
        self._V = nn.Linear(64, 1, device=self._device)
        # Advantage function
        self._A = nn.Linear(64, n_actions, device=self._device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self._device)

    def forward(self, state):
        x = self._seq(state)
        V_out = self._V(x)
        A_out = self._A(x)
        return V_out, A_out

    def get_action(self, state):
        _, advantage = self.forward(state)
        return T.argmax(advantage).item()
