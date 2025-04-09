import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_learning_curves(sac_metrics, ppo_metrics, save_path=None):
    """Plot learning curves for both algorithms"""
    plt.figure(figsize=(15, 10))

    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(
        sac_metrics["episode_rewards"], "b-", alpha=0.3, label="SAC Episode Reward"
    )
    plt.plot(sac_metrics["avg_rewards"], "b-", label="SAC Moving Average")
    plt.plot(
        ppo_metrics["episode_rewards"], "r-", alpha=0.3, label="PPO Episode Reward"
    )
    plt.plot(ppo_metrics["avg_rewards"], "r-", label="PPO Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Revenue ($)")
    plt.title("Learning Curves - Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot occupancy rates
    plt.subplot(2, 2, 2)
    plt.plot(sac_metrics["occupancy_history"], "b-", label="SAC")
    plt.plot(ppo_metrics["occupancy_history"], "r-", label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Occupancy Rate")
    plt.title("Seat Occupancy During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot actor losses
    plt.subplot(2, 2, 3)
    plt.plot(sac_metrics["actor_losses"], "b-", label="SAC Actor Loss")
    plt.plot(ppo_metrics["actor_losses"], "r-", label="PPO Actor Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Actor Loss")
    plt.title("Actor Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot critic losses
    plt.subplot(2, 2, 4)
    plt.plot(sac_metrics["critic_losses"], "b-", label="SAC Critic Loss")
    plt.plot(ppo_metrics["critic_losses"], "r-", label="PPO Critic Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Critic Loss")
    plt.title("Critic Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_training_efficiency(sac_metrics, ppo_metrics, save_path=None):
    """Plot training efficiency metrics"""
    plt.figure(figsize=(15, 5))

    # Plot training times
    plt.subplot(1, 2, 1)
    sac_cumulative_time = np.cumsum(sac_metrics["training_times"])
    ppo_cumulative_time = np.cumsum(ppo_metrics["training_times"])

    plt.plot(sac_cumulative_time, "b-", label="SAC")
    plt.plot(ppo_cumulative_time, "r-", label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Training Time (s)")
    plt.title("Training Efficiency - Cumulative Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot reward per training time
    plt.subplot(1, 2, 2)
    sac_reward_efficiency = [
        r / t for r, t in zip(sac_metrics["avg_rewards"], sac_cumulative_time)
    ]
    ppo_reward_efficiency = [
        r / t for r, t in zip(ppo_metrics["avg_rewards"], ppo_cumulative_time)
    ]

    plt.plot(sac_reward_efficiency, "b-", label="SAC")
    plt.plot(ppo_reward_efficiency, "r-", label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Training Time")
    plt.title("Training Efficiency - Reward per Time Unit")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pricing_strategies(sac_metrics, ppo_metrics, save_path=None):
    """Plot pricing strategies for both algorithms"""
    # Select a representative episode (median by revenue)
    sac_revenues = sac_metrics["total_revenues"]
    ppo_revenues = ppo_metrics["total_revenues"]

    sac_median_idx = np.argsort(sac_revenues)[len(sac_revenues) // 2]
    ppo_median_idx = np.argsort(ppo_revenues)[len(ppo_revenues) // 2]

    sac_price_path = sac_metrics["price_paths"][sac_median_idx]
    ppo_price_path = ppo_metrics["price_paths"][ppo_median_idx]

    sac_demand_path = sac_metrics["demand_paths"][sac_median_idx]
    ppo_demand_path = ppo_metrics["demand_paths"][ppo_median_idx]

    # Create time steps
    sac_time_steps = list(range(len(sac_price_path)))
    ppo_time_steps = list(range(len(ppo_price_path)))

    plt.figure(figsize=(15, 10))

    # Plot pricing strategies
    plt.subplot(2, 1, 1)
    plt.plot(sac_time_steps, sac_price_path, "b-", linewidth=2, label="SAC")
    plt.plot(ppo_time_steps, ppo_price_path, "r-", linewidth=2, label="PPO")
    plt.xlabel("Days to Departure")
    plt.ylabel("Ticket Price ($)")
    plt.title("Dynamic Pricing Strategies Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot resulting demand
    plt.subplot(2, 1, 2)
    plt.plot(sac_time_steps, sac_demand_path, "b-", linewidth=2, label="SAC Demand")
    plt.plot(ppo_time_steps, ppo_demand_path, "r-", linewidth=2, label="PPO Demand")
    plt.xlabel("Days to Departure")
    plt.ylabel("Demand")
    plt.title("Resulting Demand from Pricing Strategies")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_performance_comparison(comparison_metrics, save_path=None):
    """Plot performance comparison between SAC and PPO"""
    plt.figure(figsize=(15, 6))

    # Revenue comparison
    plt.subplot(1, 2, 1)
    algorithms = ["SAC", "PPO"]
    revenues = [comparison_metrics["sac_revenue"], comparison_metrics["ppo_revenue"]]

    bars = plt.bar(algorithms, revenues, color=["blue", "red"])

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 100,
            f"${height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Average Revenue per Flight ($)")
    plt.title("Revenue Comparison")
    plt.grid(axis="y", alpha=0.3)

    # Occupancy comparison
    plt.subplot(1, 2, 2)
    occupancies = [
        comparison_metrics["sac_occupancy"],
        comparison_metrics["ppo_occupancy"],
    ]

    bars = plt.bar(algorithms, occupancies, color=["blue", "red"])

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Average Occupancy Rate")
    plt.title("Occupancy Comparison")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
