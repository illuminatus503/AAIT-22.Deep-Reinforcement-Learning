import os
import numpy as np
import torch as T

from .replay_buffer import ReplayBuffer
from .dueling_network import DuelingDeepQNetwork


class Agent:
    def __init_adversarial_qnets(self):
        # Crea el directorio tmp, si no existe.
        os.makedirs(self._chkpt_dir, exist_ok=True)

        # Crea las redes Q1 y Q2 para Double DQN
        self._q_eval = DuelingDeepQNetwork(
            lr=self._lr,
            n_actions=self._n_actions,
            input_dim=self._input_dim,
            name="q_func_evaluator.plk",
            chkptr_dir=self._chkpt_dir,
            device=self._device,
        )

        self._q_next = DuelingDeepQNetwork(
            lr=self._lr,
            n_actions=self._n_actions,
            input_dim=self._input_dim,
            name="q_func_evnext.plk",
            chkptr_dir=self._chkpt_dir,
            device=self._device,
        )

    def __init__(
        self,
        gamma: float,
        epsilon: float,
        lr: float,
        n_actions: int,
        input_dim: int,
        batch_size: int,
        mem_size: int = int(1e6),
        eps_min: float = 1e-2,
        eps_dec: float = 5e-7,
        replace: int = int(1e3),
        chkpt_dir: str = "tmp",
        device: T.device | str = "cpu",
    ):
        # Hiperparámetros del proceso de RL
        self._gamma = gamma

        self._lr = lr

        self._eps = epsilon
        self._eps_min = eps_min
        self._eps_dec = eps_dec

        self._n_actions = n_actions

        self._input_dim = input_dim
        self._batch_size = batch_size

        self._chkpt_dir = chkpt_dir

        self._step = 0
        self._replace_target = replace

        # Selecciona el dispositivo de la simulación principal
        self._device = device

        # Replay Buffer
        self._memory = ReplayBuffer(
            mem_size=mem_size,
            input_dim=input_dim,
            device=self._device,
            alpha=self._lr,
            beta=0.4,
            kappa=0.6,
        )

        # Evaluation functions
        self.__init_adversarial_qnets()

    @property
    def eps(self):
        return self._eps

    def decrement_epsilon(self):
        if self._eps > self._eps_min:
            self._eps -= self._eps_dec

            if self._eps < self._eps_min:
                self._eps = self._eps_min

    def choose_action(self, state, is_training=False):
        if (not is_training) or (np.random.random() > self._eps):
            state = T.from_numpy(state).to(self._device)
            _, advantage = self._q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(a=self._n_actions)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self._memory.store_transition(
            T.from_numpy(state), action, reward, T.from_numpy(next_state), done
        )

    def save_models(self):
        self._q_eval.save_checkpoint()
        self._q_next.save_checkpoint()

    def load_models(self):
        self._q_eval.load_checkpoint()
        self._q_next.load_checkpoint()

    def __train(self):
        # Sample Replay buffer antes de entrenar.
        states, actions, rewards, next_states, dones, importance = self._memory.sample_buffer(
            self._batch_size
        )
        indices = np.arange(self._batch_size)  # Indices to treat

        ## Entrenamos! Solamente entrenamos Q1 (q_eval)
        ## Q2 (q_next) es una copia anterior de Q1 o random (al principio)

        # Evaluación & train
        self._q_eval.optimiser.zero_grad()
        V_s, A_s = self._q_eval.forward(states)
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        # Solo evaluación
        V_s_, A_s_ = self._q_next.forward(next_states)
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0  # If terminated state, 0 else its value

        # Vamos a intentar aproximar el q_target = R + gamma * Q2 por Q1.
        V_s_eval, A_s_eval = self._q_eval.forward(next_states)
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)

        q_target = rewards + self._gamma * q_next[indices, max_actions]

        # ! Prioritised experience replay
        error = q_target - q_pred
        error_ = T.clone(error).detach().flatten()
        self._memory.update_priorities(indices, error_)
        
        importance = importance.view((-1, 1))
        loss = T.mean(importance * error ** 2)
        # ! END Prioritised experience replay
        
        loss.backward()
        self._q_eval.optimiser.step()

    def learn(self):
        # Hay suficientes elementos en la memoria como para extraer un batch?
        if len(self._memory) < self._batch_size:
            return

        # Hay que intercambiar las redes Q1 y Q2?
        if self._step % self._replace_target == 0:
            self._q_next.load_state_dict(self._q_eval.state_dict())
        self._step += 1

        # Por cada paso de entrenamiento, decr. eps.
        self.decrement_epsilon()

        # Y, finalmente, entrenamos.
        self.__train()
