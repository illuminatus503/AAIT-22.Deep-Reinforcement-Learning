import os
import numpy as np
import torch as T

from .replay_buffer import ReplayBuffer
from .dueling_network import DuelingDeepQNetwork


class Agent:
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
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr

        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.learn_step_counter = 0
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        os.makedirs(chkpt_dir, exist_ok=True)

        self.memory = ReplayBuffer(mem_size, input_dim)

        # Evaluation functions
        self.q_eval = DuelingDeepQNetwork(
            lr=lr,
            n_actions=n_actions,
            input_dim=input_dim,
            name="LunarLander_Dueling_Double_DQN_eval",
            chkptr_dir=self.chkpt_dir,
        )
        self.q_next = DuelingDeepQNetwork(
            lr=lr,
            n_actions=n_actions,
            input_dim=input_dim,
            name="LunarLander_Dueling_Double_DQN_next",
            chkptr_dir=self.chkpt_dir,
        )

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(
                self.q_eval.device
            )
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimiser.zero_grad()

        # Do we have to replace?
        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)  # Indices to treat

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)  # Double DQN: for update rule

        # Adding together the networks
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0  # If terminated state, 0 else its value
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimiser.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
