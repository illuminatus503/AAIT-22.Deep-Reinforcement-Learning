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
    VISION_FIELD = (48, 48)

    def __define_transforms(self):
        def only_green_mask(state):
            """
            Filter only green part of image
            """
            hsv_im = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
            mask_g = cv2.inRange(hsv_im, (36, 25, 25), (70, 255, 255))

            imask_green = mask_g > 0
            green = np.zeros_like(state, np.uint8)
            green[imask_green] = state[imask_green]
            return green

        def grayscale(state):
            return cv2.cvtColor(src=state, code=cv2.COLOR_RGB2GRAY)

        def crop(state):
            # Center crop
            center = np.array(state.shape) / 2
            x = center[1] - Agent.VISION_FIELD[0] / 2
            y = center[0] - Agent.VISION_FIELD[1] / 2
            state = state[
                int(y) : int(y + Agent.VISION_FIELD[1]),
                int(x) : int(x + Agent.VISION_FIELD[0]),
            ]

            return state.astype(np.float32)

        def gaussian_blur(state):
            return cv2.GaussianBlur(src=state, ksize=(5, 5), sigmaX=0.0)

        def canny_edges(state):
            return cv2.Canny(image=state, threshold1=50, threshold2=150)

        def normalize(state):
            return cv2.normalize(
                state.astype(np.float32),
                None,
                0.0,
                1.0,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

        self._image_transforms = transforms.Compose(
            [
                transforms.Lambda(only_green_mask),
                transforms.Lambda(grayscale),
                transforms.Lambda(crop),
                # transforms.Lambda(gaussian_blur),
                # transforms.Lambda(canny_edges),
                # transforms.Lambda(normalize),
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
        self._eps_rate = EpsilonRate(n_steps=1000, init_eps=eps, min_eps=min_eps)

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
        self.__init_mem(
            mem_size=mem_size, input_dims=Agent.VISION_FIELD, batch_size=batch_size
        )
        self.__init_q_func(
            input_dims=Agent.VISION_FIELD,
            chkpt_dir=chkpt_dir,
            eps=epsilon,
            min_eps=eps_min,
        )

    @property
    def eps(self):
        return self._eps_rate.eps

    def store_transition(self, state, action, reward, next_state, done):
        self._memory.store_transition(
            T.from_numpy(self._image_transforms(state)).to(self._device),  # Solo verde
            action,
            reward,
            T.from_numpy(self._image_transforms(next_state)).to(self._device),
            done,
        )

    def save_model(self):
        self._q_funcs.save_model()

    def load_model(self):
        self._q_funcs.load_model()

    def choose_action(self, state):
        if self._eps_rate.choose_rand():
            state = T.from_numpy(self._image_transforms(state))
            state = state.view(1, 1, *state.shape)
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
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

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

        # It is a good idea to clip the gradients for image processing
        # Avoid exploding grads. 
        # https://notanymike.github.io/Solving-CarRacing/
        loss.backward()
        
        self._q_funcs.clip_grad()
        self._q_funcs.optimiser_step()

    def learn(self):
        if not self._memory.is_enough_batched():
            return

        if self._step % self._replace_target == 0:
            self._q_funcs.exchange_nets()
        self._step += 1

        self.__train()
        self._eps_rate.step_eps()
