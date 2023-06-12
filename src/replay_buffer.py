import torch as T


class ReplayBuffer:
    def __init_mems(self):
        # In Torch, default dtype=torch.float32
        self._state_memory = T.zeros(
            self._mem_size, self._input_dim, device=self.device
        )
        self._nstate_memory = T.zeros(
            self._mem_size, self._input_dim, device=self.device
        )
        self._action_memory = T.zeros(self._mem_size, dtype=T.int64, device=self.device)
        self._reward_memory = T.zeros(self._mem_size, device=self.device)
        self._terminal_memory = T.zeros(
            self._mem_size, dtype=T.bool, device=self.device
        )

    def __init__(
        self,
        mem_size: int,
        input_dim: int,
        batch_size: int,
        device: T.device | str = "cpu",
    ):
        self._input_dim = input_dim
        self.device = device

        self._mem_size = mem_size
        self._batch_size = batch_size
        self._mem_length = 0
        self._mem_idx = 0

        self.__init_mems()

    def __get_next_idx(self):
        # Revisamos cuanta memoria ha sido ocupada hasta
        # el momento: el límite es el tamaño de la memoria.
        if self._mem_length < self._mem_size:
            self._mem_length += 1

        # Revisamos si el índice de memoria ha excedido
        # los límites.
        if self._mem_idx == self._mem_size:
            self._mem_idx = 0

        mem_idx = self._mem_idx
        self._mem_idx += 1

        return mem_idx

    def store_transition(self, state, action, reward, next_state, done):
        mem_idx = self.__get_next_idx()

        self._state_memory[mem_idx] = state
        self._nstate_memory[mem_idx] = next_state
        self._reward_memory[mem_idx] = reward
        self._action_memory[mem_idx] = action
        self._terminal_memory[mem_idx] = done

    def is_enough_batched(self):
        return self._mem_idx >= self._batch_size

    def __get_batch(self):
        uniform_sample = T.ones(self._mem_length, device=self.device)

        # Sampleamos los índices del vector.
        random_sample = uniform_sample.multinomial(
            num_samples=self._batch_size, replacement=False
        )

        return random_sample

    def sample_buffer(self):
        # Obtenemos un batch aleatorio a partir de los índices
        # de memoria.
        batch_range = self.__get_batch()

        # Estos serán los valores que se devolverán.
        batch = (
            self._state_memory[batch_range],
            self._action_memory[batch_range],
            self._reward_memory[batch_range],
            self._nstate_memory[batch_range],
            self._terminal_memory[batch_range],
        )

        return batch
