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
            input_dim=self._input_dim,
            name="LunarLander_Dueling_Double_DQN_next",
            chkptr_dir=self._chkpt_dir,
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
        chkpt_dir: str = "tmp",
        device: T.device | str = "cpu",
    ):
        self._step = 0
        self._replace_target = replace

        # Selecciona el dispositivo de la simulación principal
        self._device = device

        # Replay Buffer
        self._memory = ReplayBuffer(mem_size, input_dim, self._device)

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
        return self._eps

    def decrement_epsilon(self):
        if self._eps > self._eps_min:
            self._eps -= self._eps_dec

            if self._eps < self._eps_min:
                self._eps = self._eps_min

    def choose_action(self, state):
        if np.random.random() > self._eps:
            state = T.from_numpy(state).to(self._device)
            _, advantage = self._q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(a=self._n_actions)
        return action

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
        states, actions, rewards, next_states, dones = self._memory.sample_buffer(
            self._batch_size
        )
        indices = np.arange(self._batch_size)  # Indices to treat

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

        loss = self._q_eval.loss(q_target, q_pred)
        loss.backward()
        self._q_funcs.optimiser_step()

    def learn(self):
        # Hay suficientes elementos en la memoria como para extraer un batch?
        if self._memory._mem_idx < self._batch_size:
            return

        # Hay que intercambiar las redes Q1 y Q2?
        if self._step % self._replace_target == 0:
            self._q_funcs.exchange_nets()
        self._step += 1

        # Y, finalmente, entrenamos.
        self.__train()
        self._eps_rate.step_eps()
