import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # If actions need to be clipped or squashed, do so here.
        return self.max_action * torch.tanh(x)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        # Returns only Q1
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class DPERReplayBuffer(object):

    def __init__(self, state_dim, action_dim, max_size=1e6, alpha=0.6):

        self.max_size = max_size
        self.size = 0
        self.ptr = 0
        self.alpha = alpha

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.not_dones = np.zeros((max_size, 1), dtype=np.float32)
        self.priorities = np.zeros((max_size,), dtype=np.float32) + 1e-6

    def add(self, state, action, reward, next_state, done, td_error=1.0):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.not_dones[self.ptr] = 1.0 - done
        self.priorities[self.ptr] = (abs(td_error) + 1e-6) ** self.alpha

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha

    def sample_for_critic(self, batch_size):
        if self.size == self.max_size:
            # Full buffer
            total_p = self.priorities.sum()
        else:
            total_p = self.priorities[:self.size].sum()

        probs = self.priorities[:self.size] / (total_p + 1e-6)
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Gather data
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        not_dones = torch.FloatTensor(self.not_dones[indices])
        sampling_probs = probs[indices] 
        return states, actions, rewards, next_states, not_dones, indices, sampling_probs

    def sample_for_actor_candidates(self, batch_size, K):
        all_batches = []
        for _ in range(K):
            # sample a batch at random
            indices = np.random.choice(self.size, batch_size, replace=False)
            s = torch.FloatTensor(self.states[indices])
            a = torch.FloatTensor(self.actions[indices])
            all_batches.append((s, a, indices))
        return all_batches

def compute_kl_from_exploration(mu, Sigma, sigma_exploration):

    dim = mu.shape[0]
    det_Sigma = np.linalg.det(Sigma)
    det_Sigma = max(det_Sigma, 1e-12)  # to avoid numerical blowup
    log_term = np.log((sigma_exploration**2)**dim) - np.log(det_Sigma)
    trace_term = (1.0 / (sigma_exploration**2)) * np.trace(Sigma)
    quad_term = (1.0 / (sigma_exploration**2)) * (mu @ mu)

    kl = 0.5 * (log_term + trace_term + quad_term - dim)
    return kl

class DPER_TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        sigma_exploration=0.2,
        lr=3e-4,
        buffer_size=1_000_000,
        IS=False,
        beta_start=0.4,
        beta_frames=1_000_000,
    ):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.sigma_exploration = sigma_exploration

        self.total_it = 0
        self.IS=IS
        self.beta_start  = beta_start
        self.beta_frames = beta_frames    
        # Decoupled PER buffer
        self.replay_buffer = DPERReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    def select_action(self, state):
        # Convert to torch and get action from actor
        state_t = torch.FloatTensor(state)
        action = self.actor(state_t).cpu().data.numpy().flatten()
        return action

    def train(self, batch_size=256, K=4):

        self.total_it += 1
        (states, actions, rewards, next_states, not_dones, indices, sampling_probs
        ) = self.replay_buffer.sample_for_critic(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = self.actor_target(next_states) + noise
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_dones * self.gamma * target_Q
        
        if self.IS:
            fraction = min(float(self.total_it) / self.beta_frames, 1.0)
            beta = self.beta_start + fraction * (1.0 - self.beta_start)
            current_Q1, current_Q2 = self.critic(states, actions)
            N    = float(self.replay_buffer.size)
            # importance weights
            w  = (1.0 / (N * sampling_probs)) ** beta            # shape: (batch,)
            w  = w / w.max()                                     # normalize so max weight = 1
            w  = torch.from_numpy(w).unsqueeze(1)    # shape: (batch,1)

            # now per‐sample TD errors
            td1 = (current_Q1 - target_Q).pow(2)                  # (batch,1)
            td2 = (current_Q2 - target_Q).pow(2)

            # weighted MSE
            loss1 = (w * td1).mean()
            loss2 = (w * td2).mean()
            critic_loss = loss1 + loss2
        
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            new_td_errors = (current_Q1 - target_Q).abs() + 1e-6
            new_td_errors = new_td_errors.cpu().data.numpy().flatten()
        self.replay_buffer.update_priorities(indices, new_td_errors)

        if self.total_it % self.policy_freq == 0:
            candidate_batches = self.replay_buffer.sample_for_actor_candidates(
                batch_size, K
            )
            best_kl = float("inf")
            best_batch = None
            for (s, a, indices_batch) in candidate_batches:
                with torch.no_grad():
                    a_current = self.actor(s).cpu()  # shape [batch_size, action_dim]
                x_dot = a_current.numpy() - a.numpy()  # shape [batch_size, action_dim]
                mu = np.mean(x_dot, axis=0)  # shape [action_dim,]
                x_centered = x_dot - mu
                Sigma = np.cov(x_centered.T, bias=False)  # shape [action_dim, action_dim]
                kl = compute_kl_from_exploration(mu, Sigma, self.sigma_exploration)

                if kl < best_kl:
                    best_kl = kl
                    best_batch = (s, a, indices_batch)

            if best_batch is not None:
                s_best, _, _ = best_batch
                actor_loss = -self.critic.Q1(s_best, self.actor(s_best)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done, td_error=1.0)

