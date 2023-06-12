import os
import torch as T
import torch.nn as nn


class DoubleDQN(nn.Module):
    def __init_qnets(self, dqn_type):
        # Q1 network
        self._eval_chkpoint_file = os.path.join(self._chkpoint_path, "eval_net.plk")
        self._q_eval = dqn_type(
            lr=self._lr,
            n_actions=self._n_actions,
            input_dim=self._input_dims,
            chkpoint_file=self._eval_chkpoint_file,
            device=self._device,
        )

        # Q2 network
        self._next_chkpoint_file = os.path.join(self._chkpoint_path, "next_net.plk")
        self._q_next = dqn_type(
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
        dqn_type,
        chkpoint_path: str = "tmp/double_dqn",
        device: T.device | str = "cpu",
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
        self.__init_qnets(dqn_type)
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
