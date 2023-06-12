import torch as T

from .replay_buffer import ReplayBuffer
from .deep_q_net import DoubleDQN


class EpsilonRate:
    def __init__(self, init_eps=1.0, min_eps=0.01, decr_rate=0.001):
        self._eps = init_eps
        self._min_eps = min_eps
        self._decr_rate = decr_rate

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
    def __init_mem(self, mem_size, input_dim, batch_size):
        self._batch_idx = T.arange(batch_size, dtype=T.int32, device=self._device)

        # Replay Buffer
        self._memory = ReplayBuffer(
            mem_size=mem_size,
            input_dim=input_dim,
            batch_size=batch_size,
            device=self._device,
        )

    def __init_q_func(self, input_dim, chkpt_dir, eps, min_eps, decr_rate):
        self._chkpt_dir = chkpt_dir

        # Evaluation functions
        self._q_funcs = DoubleDQN(
            input_dims=input_dim,
            lr=self._lr,
            n_actions=self._n_actions,
            device=self._device,
        )

        # Epsilon rate
        self._eps_rate = EpsilonRate(init_eps=eps, min_eps=min_eps, decr_rate=decr_rate)

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
        self.__init_mem(mem_size=mem_size, input_dim=input_dim, batch_size=batch_size)
        self.__init_q_func(
            input_dim=input_dim,
            chkpt_dir=chkpt_dir,
            eps=epsilon,
            min_eps=eps_min,
            decr_rate=eps_dec,
        )

    @property
    def eps(self):
        return self._eps_rate.eps

    def store_transition(self, state, action, reward, next_state, done):
        state_ = T.from_numpy(state).to(self._device)
        next_state_ = T.from_numpy(next_state).to(self._device)
        self._memory.store_transition(state_, action, reward, next_state_, done)

    def save_model(self):
        self._q_funcs.save_model()

    def load_model(self):
        self._q_funcs.load_model()

    def choose_action(self, state):
        if self._eps_rate.choose_rand():
            return self._q_funcs.get_action(T.from_numpy(state).to(self._device))

        return T.randint(0, self._n_actions, (1,)).item()

    def __train(self):
        # Sample Replay buffer antes de entrenar.
        (
            states,
            actions,
            rewards,
            next_states,
            terminal_states,
        ) = self._memory.sample_buffer()

        ## Entrenamos! Solamente entrenamos Q1 (q_eval)
        ## Q2 (q_next) es una copia anterior de Q1 o random (al principio)

        # Evaluación & train
        self._q_funcs.zero_grad()

        # Q1: seleccionamos los Q-valores
        q_eval = self._q_funcs.forward(states)[self._batch_idx, actions]

        # Q2: seleccionamos las acciones
        q_next = self._q_funcs.next_forward(next_states)
        q_next[terminal_states] = 0.0

        # ESTO es lo que queremos que suceda
        q_target = rewards + self._gamma * T.max(q_next, dim=1)[0]

        loss = self._q_funcs.loss(q_target, q_eval)
        loss.backward()
        self._q_funcs.optimiser_step()

    def learn(self):
        if not self._memory.is_enough_batched():
            return

        # Hay que intercambiar las redes Q1 y Q2?
        if self._step % self._replace_target == 0:
            self._q_funcs.exchange_nets()
        self._step += 1

        # Y, finalmente, entrenamos.
        self.__train()
        self._eps_rate.step_eps()
