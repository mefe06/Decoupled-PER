import argparse
import gymnasium as gym
import numpy as np
import torch
import random
import sys
import time
def evaluate_policy(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.action_space.seed(args.seed)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False

        while not done:
            action = agent.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------", file=sys.stderr)
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}", file=sys.stderr)
    print("---------------------------------------", file=sys.stderr)

    return avg_reward

# -------------------------------
# Simple Uniform Replay Buffer
# -------------------------------
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

# -------------------------------
# Main Training Loop
# -------------------------------
def main(args):
    # Create environment
    env = gym.make(args.env_name)
    env.action_space.seed(args.seed)
    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # -------------------------------
    # Instantiate Agent
    # -------------------------------
    if args.replay == "DPER":
        from DPER_TD3 import DPER_TD3
        agent = DPER_TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            gamma=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            sigma_exploration=args.sigma_exploration,
            lr=args.lr,
        )
    elif args.replay == "PER":
        from PER_TD3 import PER_TD3 
        agent = PER_TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=args.device,
            discount=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
        )
        replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    elif args.replay == "ER":
        from TD3 import TD3 as ER_TD3
        agent = ER_TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=args.device,
            discount=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
        )
        replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    elif args.replay == "uniform_DPER":
        from DPER_uniform_critic_TD3 import DPER_TD3
        agent = DPER_TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            gamma=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            sigma_exploration=args.sigma_exploration,
            lr=args.lr,
        )
    else:
        raise ValueError("Unknown replay type. Choose from DPER, PER, or ER.")
    
    # -------------------------------
    # Training Hyperparameters
    # -------------------------------
    total_timesteps = args.total_timesteps
    start_timesteps = args.start_timesteps
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    state, _ = env.reset()
    evaluations = [evaluate_policy(agent, args.env_name, args.seed)]
    training_start = time.time()
    eval_time_accum = 0.0
    t0 = time.time()
    evaluations = [evaluate_policy(agent, args.env_name, args.seed)]
    eval_time_accum += time.time() - t0
    for t in range(int(total_timesteps)):
        episode_timesteps += 1        
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array([state]))
            action = (action + np.random.normal(0, args.expl_noise, size=action_dim)).clip(-max_action, max_action)
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        if args.replay in ["DPER", "uniform_DPER", "PER"]:
            agent.add_to_replay_buffer(state, action, reward, next_state, done_bool)
        else:
            replay_buffer.add(state, action, reward, next_state, done_bool)
            
        state = next_state
        episode_reward += reward
        
        # Train agent once enough data has been collected
        if t >= start_timesteps:
            if args.replay in ["DPER", "uniform_DPER"]:
                agent.train(batch_size=args.batch_size, K=args.K)
            elif args.replay == "PER":
                agent.update_parameters(batch_size=args.batch_size)
            else:
                agent.update_parameters(replay_buffer, batch_size=args.batch_size)
                
        if done:
            print(f"Total T: {t+1} | Episode Num: {episode_num+1} | Episode T: {episode_timesteps} | Reward: {episode_reward:.3f}", file=sys.stderr)
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        file_name = f"{args.replay}_{args.K}_{args.env_name}_{args.seed}"
        if (t + 1) % args.eval_freq == 0:
            t_eval_start = time.time()
            evaluations.append(evaluate_policy(agent, args.env_name, args.seed))
            eval_time_accum += time.time() - t_eval_start
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                agent.save(f"./models/{file_name}")
    total_time = time.time() - training_start
    train_only = total_time - eval_time_accum
    print(f"ELAPSED_TIME_EXCL_EVAL:{train_only:.2f}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v5", help="Gym environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--replay", type=str, default="uniform_DPER", choices=["DPER", "PER", "ER", "uniform_DPER"],
                        help="Experience replay type to use: DPER, PER, or ER")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--start_timesteps", type=int, default=25000,
                        help="Time steps initial random policy is used")
    parser.add_argument("--expl_noise", type=float, default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--K", type=int, default=4,
                        help="Number of candidate batches for DPER actor update (only used in DPER)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", type=int, default=2, help="Frequency of delayed policy updates")
    parser.add_argument("--sigma_exploration", type=float, default=0.2,
                        help="Exploration noise scale for DPER (only used in DPER)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (used by DPER)")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Max size of the replay buffer")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--eval_freq", type=int, default=1e3, help="Evaluation period in number of time steps")
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    args = parser.parse_args()
    
    main(args)
