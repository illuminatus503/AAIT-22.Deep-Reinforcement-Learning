import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.models import resnet18


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
        self._transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._resnet = resnet18(pretrained=True)

        # Freeze RESNET
        # Congelar los pesos de todas las capas excepto la última capa lineal
        for name, param in self._resnet.named_parameters():
            if name != "fc.weight" and name != "fc.bias":
                param.requires_grad = False

        # Value function
        self._V = nn.Linear(10, 1, device=self._device)
        # Advantage function
        self._A = nn.Linear(10, n_actions, device=self._device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self._device)

    def forward(self, state):
        transformed = self._transforms(state)
        out = self._resnet.forward(transformed)
        V_out = self._V(out)
        A_out = self._A(out)
        return V_out, A_out

    def get_action(self, state):
        _, advantage = self.forward(state)
        return T.argmax(advantage).item()
