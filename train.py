import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import AirlinePricingEnv
from sac_agent import SAC
from ppo_agent import PPO
from tqdm import tqdm
import os
import time


def train_agent(agent_type, data, episodes=500, save_dir="models"):
    """Train an agent (SAC or PPO) for airline pricing"""

    # Create environment
    env = AirlinePricingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create agent
    if agent_type == "SAC":
        agent = SAC(state_size, action_size, max_action)
    elif agent_type == "PPO":
        agent = PPO(state_size, action_size, max_action)
    else:
        raise ValueError("agent_type must be 'SAC' or 'PPO'")

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training metrics
    episode_rewards = []
    avg_rewards = []
    revenue_history = []
    occupancy_history = []
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    training_times = []

    # Training loop
    for episode in tqdm(range(episodes), desc=f"Training {agent_type}"):
        episode_start_time = time.time()
        state = env.reset()
        total_reward = 0
        done = False

        episode_data = {"prices": [], "demands": [], "revenues": [], "steps": []}

        while not done:
            # Select action
            if agent_type == "SAC":
                action = agent.select_action(state)
            else:  # PPO
                action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store transition
            if agent_type == "SAC":
                agent.replay_buffer.add(state, action, reward, next_state, done)
            else:  # PPO
                agent.store_transition(reward, done)

            # Record step data
            episode_data["prices"].append(env.current_price)
            episode_data["demands"].append(env.recent_demand)
            episode_data["revenues"].append(reward)
            episode_data["steps"].append(env.current_step)

            # Move to next state
            state = next_state
            total_reward += reward

            # Train agent
            if agent_type == "SAC":
                critic_loss, actor_loss, alpha_loss = agent.train()
                if (
                    episode % 10 == 0
                ):  # Record loss every 10 episodes to reduce overhead
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    entropy_losses.append(alpha_loss)

        # For PPO, train after episode is complete
        if agent_type == "PPO":
            actor_loss, critic_loss, entropy_loss = agent.train()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_losses.append(entropy_loss)

        # Record episode time
        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time
        training_times.append(episode_time)

        # Save model every 100 episodes
        if episode % 100 == 0:
            if agent_type == "SAC":
                agent.save(f"{save_dir}/{agent_type}_agent_ep{episode}")
            else:  # PPO
                agent.save(f"{save_dir}/{agent_type}_agent_ep{episode}")

        # Record episode metrics
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        # Calculate occupancy
        occupancy = (180 - env.seats_remaining) / 180
        occupancy_history.append(occupancy)
        revenue_history.append(total_reward)

        # Log progress
        if episode % 10 == 0:
            print(
                f"Episode: {episode}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Occupancy: {occupancy:.2f}"
            )

    # Save final model
    if agent_type == "SAC":
        agent.save(f"{save_dir}/{agent_type}_agent_final")
    else:  # PPO
        agent.save(f"{save_dir}/{agent_type}_agent_final")

    # Return training metrics
    return {
        "episode_rewards": episode_rewards,
        "avg_rewards": avg_rewards,
        "revenue_history": revenue_history,
        "occupancy_history": occupancy_history,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_losses": entropy_losses,
        "training_times": training_times,
    }
