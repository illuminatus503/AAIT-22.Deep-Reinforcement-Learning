import torch as T


class ReplayBuffer:
    def __init_mems(self):
        # In Torch, default dtype=torch.float32
        self._state_memory = T.zeros(
            self._mem_size, self._input_dim, device=self._device
        )
        self._nstate_memory = T.zeros(
            self._mem_size, self._input_dim, device=self._device
        )
        self._action_memory = T.zeros(
            self._mem_size, dtype=T.int64, device=self._device
        )
        self._reward_memory = T.zeros(self._mem_size, device=self._device)
        self._terminal_memory = T.zeros(
            self._mem_size, dtype=T.bool, device=self._device
        )

    def __init__(
        self,
        mem_size: int,
        input_dim: int,
        alpha: float,
        beta: float,
        kappa: float,
        device: T.device | str = "cpu",
        eps: float = 1e-2,
    ):
        self._input_dim = input_dim
        self._device = device

        self._mem_size = mem_size
        self._mem_length = 0

        self.__init_mems()

        # Prioritised Experience Replay
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._eps = eps

        self._priorities = T.zeros(self._mem_size, device=self._device)

    def __get_mem_idx(self):
        mem_idx = int(T.argmin(self._priorities).item())

        if self._mem_length < self._mem_size:
            self._mem_length += 1

        return mem_idx

    def store_transition(self, state, action, reward, next_state, done):
        mem_idx = self.__get_mem_idx()

        self._state_memory[mem_idx] = state
        self._nstate_memory[mem_idx] = next_state
        self._reward_memory[mem_idx] = reward
        self._action_memory[mem_idx] = action
        self._terminal_memory[mem_idx] = done

        # Actualizamos la prioridad para el índice dado
        self._priorities[mem_idx] = max(1.0, T.max(self._priorities).item())

    def _get_priority(self, error):
        return (T.abs(error) + self._eps) ** self._alpha

    def update_priorities(self, idx, error):
        self._priorities[idx] = self._get_priority(error)

    def __get_probs(self):
        likelyhood = (
            T.clone(self._priorities[: self._mem_length]).detach() ** self._kappa
        )
        probs = likelyhood / T.sum(likelyhood).item()
        return probs

    def __get_batch(self, probs, batch_size: int):
        random_sample = probs.multinomial(
            num_samples=batch_size,
            replacement=False,
        )

        return random_sample

    def __get_importance(self, probs):
        importance = (self._mem_length * probs) ** -self._beta
        importance_n = importance / T.max(importance).item()
        return importance_n

    def sample_buffer(self, batch_size: int):
        # Obtenemos un batch aleatorio a partir de los índices
        # de memoria.
        sample_probabilities = self.__get_probs()
        batch_range = self.__get_batch(sample_probabilities, batch_size)

        # Estos serán los valores que se devolverán.
        batch = (
            self._state_memory[batch_range],
            self._action_memory[batch_range],
            self._reward_memory[batch_range],
            self._nstate_memory[batch_range],
            self._terminal_memory[batch_range],
            self.__get_importance(sample_probabilities[batch_range]),
        )

        return batch
    
    def __len__(self):
        return self._mem_length    
