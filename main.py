import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from environment import AirlinePricingEnv
from sac_agent import SAC
from ppo_agent import PPO
from train import train_agent
from evaluate import evaluate_agent, compare_algorithms
from visualize import (
    plot_learning_curves,
    plot_training_efficiency,
    plot_pricing_strategies,
    plot_performance_comparison,
)


def main():
    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    # Load and preprocess data from Kaggle
    print("Loading flight data...")
    # For this example, we're simulating data
    # In practice, you would load from Kaggle dataset
    data = simulate_flight_data()

    # Create environment
    env = AirlinePricingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Train SAC agent
    print("\nTraining SAC agent...")
    sac_training_metrics = train_agent("SAC", data, episodes=500, save_dir="models")

    # Train PPO agent
    print("\nTraining PPO agent...")
    ppo_training_metrics = train_agent("PPO", data, episodes=500, save_dir="models")

    # Plot learning curves
    plot_learning_curves(
        sac_training_metrics,
        ppo_training_metrics,
        save_path="results/learning_curves.png",
    )

    # Plot training efficiency
    plot_training_efficiency(
        sac_training_metrics,
        ppo_training_metrics,
        save_path="results/training_efficiency.png",
    )

    # Create and load agents for evaluation
    sac_agent = SAC(state_size, action_size, max_action)
    sac_agent.load("models/SAC_agent_final")

    ppo_agent = PPO(state_size, action_size, max_action)
    ppo_agent.load("models/PPO_agent_final")

    # Evaluate agents
    print("\nEvaluating SAC agent...")
    sac_eval_metrics = evaluate_agent("SAC", sac_agent, env, episodes=50)

    print("\nEvaluating PPO agent...")
    ppo_eval_metrics = evaluate_agent("PPO", ppo_agent, env, episodes=50)

    # Compare algorithms
    comparison_metrics = compare_algorithms(sac_eval_metrics, ppo_eval_metrics)

    # Plot pricing strategies
    plot_pricing_strategies(
        sac_eval_metrics, ppo_eval_metrics, save_path="results/pricing_strategies.png"
    )

    # Plot performance comparison
    plot_performance_comparison(
        comparison_metrics, save_path="results/performance_comparison.png"
    )

    print("\nTraining and evaluation complete. Results saved to 'results' directory.")


def simulate_flight_data():
    """Generate simulated flight data for demonstration"""
    # In a real implementation, this would load data from Kaggle
    np.random.seed(42)
    n_flights = 100

    data = {
        "flight_id": range(1, n_flights + 1),
        "base_demand": np.random.normal(100, 20, n_flights),
        "base_price": np.random.normal(200, 30, n_flights),
        "elasticity": np.random.uniform(1.0, 2.0, n_flights),
        "capacity": np.random.choice([150, 180, 220], n_flights),
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
