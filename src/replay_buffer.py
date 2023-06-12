import torch as T
import torch.nn.functional as F


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

        self._priorities = T.zeros(self._mem_size, device=self._device)

    def __init__(
        self,
        input_dim: int,
        alpha: float,
        mem_size: int = int(1e5),
        batch_size: int = 32,
        beta: float = 0.4,
        kappa: float = 0.6,
        device: T.device | str = "cpu",
    ):
        self._input_dim = input_dim
        self._device = device

        # Hyperparameters
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._eps = 1e-6  # For numerical stability

        self._mem_size = mem_size
        self._batch_size = batch_size

        # Initialise memory
        self._mem_length = 0
        self.__init_mems()

    def is_enough_batched(self):
        return self._mem_length >= self._batch_size

    def __get_mem_idx(self):
        mem_idx = int(T.argmin(self._priorities).item())
        if self._mem_length < self._mem_size:
            self._mem_length += 1
        return mem_idx

    def __get_priority(self):
        priority = T.max(self._priorities).item()
        if priority > 1.0:
            return priority
        return 1.0

    def store_transition(self, state, action, reward, next_state, done):
        mem_idx = self.__get_mem_idx()

        self._state_memory[mem_idx] = state
        self._nstate_memory[mem_idx] = next_state
        self._reward_memory[mem_idx] = reward
        self._action_memory[mem_idx] = action
        self._terminal_memory[mem_idx] = done
        self._priorities[mem_idx] = self.__get_priority()

    def __sample_experience(self):
        return F.normalize(
            self._priorities[: self._mem_length] ** self._kappa, p=1.0, dim=0
        )

    def __get_batch(self, probs):
        random_sample = probs.multinomial(
            num_samples=self._batch_size,
            replacement=False,
        )
        return random_sample

    def __compute_importance_sampling(self, probs):
        importance = (self._mem_length * probs) ** -self._beta
        importance_n = importance / T.max(importance).item()
        return importance_n

    def sample_buffer(self):
        probabilities = self.__sample_experience()
        batch_range = self.__get_batch(probabilities)
        return (
            self._state_memory[batch_range],
            self._action_memory[batch_range],
            self._reward_memory[batch_range],
            self._nstate_memory[batch_range],
            self._terminal_memory[batch_range],
            self.__compute_importance_sampling(probabilities[batch_range]),
        )

    def update_experiences(self, idx, error):
        # self._priorities[idx] = (T.abs(error) + self._eps) ** self._alpha
        self._priorities[idx] = T.abs(error)  # Like in Rainbow
