import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        ## Arquitectura
        # Capa principal: detección inicial
        self._fc1 = nn.Linear(input_dim, 512, device=self._device)
        # Value function
        self._V = nn.Linear(512, 1, device=self._device)
        # Advantage function
        self._A = nn.Linear(512, n_actions, device=self._device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self._device)

    def forward(self, state):
        flat1 = F.relu(self._fc1(state))
        V_out = self._V(flat1)
        A_out = self._A(flat1)
        return V_out, A_out

    def get_action(self, state):
        _, advantage = self.forward(state)
        return T.argmax(advantage).item()
