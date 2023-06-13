import torch as T

from .replay_buffer import ReplayBuffer
from .double_dqn import DoubleDQN
from .deep_q_net import DuelingDeepQNetwork

from torchvision import transforms
import numpy as np
import cv2


class EpsilonRate:
    def __init__(self, n_steps, init_eps=1.0, min_eps=0.01):
        self._eps = init_eps
        self._min_eps = min_eps
        self._decr_rate = (init_eps - min_eps) / n_steps

    @property
    def eps(self):
        return self._eps

    @property
    def min_eps(self):
        return self._min_eps

    @property
    def eps_decr_rate(self):
        return self._decr_rate

    def step_eps(self):
        if self._eps > self._min_eps:
            self._eps -= self._decr_rate

            if self._eps < self._min_eps:
                self._eps = self._decr_rate

    def choose_rand(self):
        return T.rand(1).item() > self._eps


class Agent:
    def __define_transforms(self):
        def only_green_mask(state):
            """
            Filter only green part of image
            """
            state = np.array(state.cpu())
            hsv_im = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
            mask_g = cv2.inRange(hsv_im, (36, 25, 25), (70, 255, 255))

            imask_green = mask_g > 0
            green = np.zeros_like(state, np.uint8)
            green[imask_green] = state[imask_green]
            return green

        def canny_edges(state):
            return cv2.Canny(np.array(state.cpu()), 50, 150)

        def set_format(state):
            state = state.cpu().to(T.float32)
            n = len(state.shape)
            if n == 3:
                state = state.unsqueeze(0)
            return state.permute(0, n - 1, *list(range(1, n - 1)))

        self._transforms = transforms.Compose(
            [
                # transforms.Lambda(only_green_mask),
                # transforms.Lambda(set_format),
                transforms.Grayscale(),
                transforms.GaussianBlur(kernel_size=5),
                # transforms.Lambda(canny_edges),
                # transforms.ToTensor(),
                transforms.Normalize(mean=0.0, std=1.0),
            ]
        )

    def __init_mem(self, mem_size, input_dims, batch_size):
        self._batch_idx = T.arange(batch_size, dtype=T.int32, device=self._device)

        # Replay Buffer
        self._memory = ReplayBuffer(
            mem_size=mem_size,
            input_dims=input_dims,
            alpha=self._lr,
            batch_size=batch_size,
            device=self._device,
        )

    def __init_q_func(self, input_dims, chkpt_dir, eps, min_eps):
        self._chkpt_dir = chkpt_dir

        # Evaluation functions
        self._q_funcs = DoubleDQN(
            input_dims=input_dims,
            lr=self._lr,
            n_actions=self._n_actions,
            device=self._device,
            dqn_type=DuelingDeepQNetwork,
        )

        # Epsilon rate
        self._eps_rate = EpsilonRate(n_steps=100, init_eps=eps, min_eps=min_eps)

    def __init__(
        self,
        lr: float,
        gamma: float,
        input_dims: int,
        n_actions: int,
        epsilon: float = 1.0,
        eps_min: float = 0.01,
        batch_size: int = 32,
        mem_size: int = int(1e5),
        replace: int = int(1e3),
        chkpt_dir: str = "tmp/dueling_ddqn",
        device: T.device | str = "cpu",
    ):
        self._step = 0
        self._replace_target = replace

        # Selecciona el dispositivo de la simulación principal
        self._device = device

        # Hiperparámetros
        self._lr = lr
        self._gamma = gamma
        self._n_actions = n_actions

        # Q-Function(s)
        self.__define_transforms()  # Transformamos la imagen a 96x96x1
        # No alteramos el tamaño de la memoria para evitar overhead por transformaciones
        self.__init_mem(mem_size=mem_size, input_dims=input_dims, batch_size=batch_size)
        self.__init_q_func(
            input_dims=input_dims[:2],
            chkpt_dir=chkpt_dir,
            eps=epsilon,
            min_eps=eps_min,
        )

    @property
    def eps(self):
        return self._eps_rate.eps

    def store_transition(self, state, action, reward, next_state, done):
        self._memory.store_transition(
            T.from_numpy(state).to(T.float32).to(self._device),
            action,
            reward,
            T.from_numpy(next_state).to(T.float32).to(self._device),
            done,
        )

    def save_model(self):
        self._q_funcs.save_model()

    def load_model(self):
        self._q_funcs.load_model()

    def choose_action(self, state):
        if self._eps_rate.choose_rand():
            state = T.from_numpy(state).to(T.float32)
            state = state.permute(2, 0, 1)
            state = state.view(1, *state.shape)
            state = self._transforms(state)
            return self._q_funcs.get_action(state.to(self._device))

        return T.randint(0, self._n_actions, (1,)).item()

    def __train(self):
        # Sample Replay buffer antes de entrenar.
        (
            states,
            actions,
            rewards,
            next_states,
            terminal_states,
            importance,
        ) = self._memory.sample_buffer()

        ## Entrenamos! Solamente entrenamos Q1 (q_eval)
        ## Q2 (q_next) es una copia anterior de Q1 o random (al principio)

        # Evaluación & train
        self._q_funcs.zero_grad()

        # Transform input states
        states = self._transforms(states.permute(0, 3, 1, 2))
        next_states = self._transforms(next_states.permute(0, 3, 1, 2))
        # states = self._transforms(states)
        # next_states = self._transforms(next_states)

        # Q1: seleccionamos los Q-valores
        V_s, A_s = self._q_funcs.forward(states)
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[
            self._batch_idx, actions
        ]

        # Q2: seleccionamos las acciones
        V_ns, A_ns = self._q_funcs.next_forward(next_states)
        q_next = T.add(V_ns, (A_ns - A_ns.mean(dim=1, keepdim=True)))
        q_next[terminal_states] = 0.0

        # ESTO es lo que queremos que suceda
        V_s_eval, A_s_eval = self._q_funcs.forward(next_states)
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        q_target = (
            rewards + self._gamma * q_next[self._batch_idx, T.argmax(q_eval, dim=1)]
        )

        # ! Prioritised experience replay
        error = q_target - q_pred
        self._memory.update_experiences(
            self._batch_idx, T.clone(error).detach().flatten()
        )
        loss = T.mean(importance.view((-1, 1)) * error**2)
        # ! END Prioritised experience replay

        loss.backward()
        self._q_funcs.optimiser_step()

    def learn(self):
        if not self._memory.is_enough_batched():
            return

        if self._step % self._replace_target == 0:
            self._q_funcs.exchange_nets()
        self._step += 1

        self.__train()
        self._eps_rate.step_eps()
