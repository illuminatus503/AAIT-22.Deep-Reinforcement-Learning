import numpy as np


class ReplayBuffer:
    def __init__(self, mem_size: int, input_dim: int):
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((mem_size, input_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((mem_size, input_dim), dtype=np.float32)
        self.action_memory = np.zeros(mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # TODO: prioritised experience replay

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
