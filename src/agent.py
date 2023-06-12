import torch as T

from .replay_buffer import ReplayBuffer
from .double_dqn import DoubleDQN
from .deep_q_net import DuelingDeepQNetwork


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

    def __init_q_func(self, input_dims, chkpt_dir, eps, min_eps, decr_rate):
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
        self._eps_rate = EpsilonRate(init_eps=eps, min_eps=min_eps, decr_rate=decr_rate)

    def __init__(
        self,
        lr: float,
        gamma: float,
        input_dims: int,
        n_actions: int,
        epsilon: float = 1.0,
        eps_min: float = 0.01,
        eps_dec: float = 1e-4,
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
        self.__init_mem(mem_size=mem_size, input_dims=input_dims, batch_size=batch_size)
        self.__init_q_func(
            input_dims=input_dims,
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
            importance,
        ) = self._memory.sample_buffer()

        ## Entrenamos! Solamente entrenamos Q1 (q_eval)
        ## Q2 (q_next) es una copia anterior de Q1 o random (al principio)

        # Evaluación & train
        self._q_funcs.zero_grad()

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
