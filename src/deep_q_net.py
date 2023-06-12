import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(
        self,
        lr: float,
        n_actions: int,
        input_dim: int,
        chkpoint_file: str,
        device: T.device | str,
    ):
        super(DeepQNetwork, self).__init__()
        self.device = device
        self.checkpoint_file = chkpoint_file

        ## Arquitectura
        # Capa principal: detección inicial
        self.fc1 = nn.Linear(input_dim, 512, device=self.device)
        self.fc2 = nn.Linear(512, 512, device=self.device)
        # Q_function
        self.Q = nn.Linear(512, n_actions, device=self.device)

        # Optimizador por defecto para ambas redes
        self.optimiser = optim.Adam(self.parameters(), lr)
        # Función de pérdida
        self.loss = nn.MSELoss()

        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        
        # Obtenemos las acciones
        actions = self.Q(flat2)
        
        return actions

    def get_action(self, state):
        actions = self.forward(state)
        return T.argmax(actions).item()

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class DoubleDQN(nn.Module):
    def __init_qnets(self):
        # Q1 network
        self._eval_chkpoint_file = os.path.join(self._chkpoint_path, 'eval_net.plk')
        self._q_eval = DeepQNetwork(
            lr=self._lr,
            n_actions=self._n_actions,
            input_dim=self._input_dims,
            chkpoint_file=self._eval_chkpoint_file,
            device=self._device,
        )
        
        # Q2 network
        self._next_chkpoint_file = os.path.join(self._chkpoint_path, 'next_net.plk')
        self._q_next = DeepQNetwork(
            lr=self._lr,
            n_actions=self._n_actions,
            input_dim=self._input_dims,
            chkpoint_file=self._next_chkpoint_file,
            device=self._device,
        )
    
    def __init__(
        self,
        lr: float,
        input_dims: int,
        n_actions: int,
        chkpoint_path:str = 'tmp/double_dqn',
        device: T.device | str = 'cpu',
    ):
        super(DoubleDQN, self).__init__()
        self._device = device
        
        # Hyperparameters
        self._lr = lr
        self._input_dims = input_dims
        self._n_actions = n_actions
        
        # Checkpoints
        self._chkpoint_path = chkpoint_path
        os.makedirs(self._chkpoint_path, exist_ok=True)
        
        # Create Q1 and Q2 nets
        self.__init_qnets()
        self.to(self._device)        
    
    def exchange_nets(self):
        self._q_next.load_state_dict(self._q_eval.state_dict())
    
    def get_action(self, state):
        return self._q_eval.get_action(state)
    
    def forward(self, state):
        return self._q_eval(state)
    
    def next_forward(self, next_state):
        return self._q_next(next_state)        
    
    def save_model(self):
        self._q_eval.save_checkpoint()
        self._q_next.save_checkpoint()
        
    def load_model(self):
        self._q_eval.load_checkpoint()
        self._q_next.load_checkpoint()
    
    def zero_grad(self):
        self._q_eval.optimiser.zero_grad()
    
    def loss(self, target, prediction):
        return self._q_eval.loss(target, prediction)
    
    def optimiser_step(self):
        self._q_eval.optimiser.step()
                    