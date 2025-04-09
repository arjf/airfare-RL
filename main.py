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
    # Load the Delhi-Mumbai flight dataset
    flight_data = load_airline_dataset()

    if flight_data is None:
        print("Error: Could not load dataset. Exiting.")
        return

    # Print dataset information
    print(f"Loaded {len(flight_data)} flight records")
    print(
        f"Price range: ₹{flight_data['price'].min()} to ₹{flight_data['price'].max()}"
    )
    print(f"Airlines: {flight_data['airline'].unique()}")

    # Create environment with the real dataset
    env = AirlinePricingEnv(flight_data)

    # Train SAC agent
    print("\nTraining SAC agent...")
    sac_agent = SAC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    )
    sac_metrics = train_agent(sac_agent, env, episodes=500)

    # Train PPO agent
    print("\nTraining PPO agent...")
    ppo_agent = PPO(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    )
    ppo_metrics = train_agent(ppo_agent, env, episodes=500)

    # Compare performance
    compare_agents(sac_metrics, ppo_metrics, env)


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


def load_airline_dataset():
    """Load and preprocess the Delhi-Mumbai flight pricing dataset"""
    import pandas as pd
    import os

    # Set up data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_path = os.path.join(data_dir, "Clean_Dataset.csv")

    # Handle dataset availability
    if not os.path.exists(dataset_path):
        # Copy from current directory if available
        if os.path.exists("Clean_Dataset.csv"):
            import shutil

            shutil.copy("Clean_Dataset.csv", dataset_path)
            print("Dataset copied to data directory")
        else:
            print("Dataset not found")
            return None

    # Load and preprocess the dataset
    df = pd.read_csv(dataset_path)

    # Convert categorical variables to numeric for RL algorithms
    time_categories = [
        "Early_Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Late_Night",
    ]
    time_map = {cat: i for i, cat in enumerate(time_categories)}

    df["departure_time_num"] = df["departure_time"].map(time_map)
    df["arrival_time_num"] = df["arrival_time"].map(time_map)

    # Map stop categories
    df["stops_num"] = df["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})

    # Encode airlines
    df["airline_code"] = pd.factorize(df["airline"])[0]

    # Convert duration to float if needed
    df["duration"] = df["duration"].astype(float)

    return df


def simulate_flight_data():
    """Generate simulated flight data for demonstration"""
    print("Generating simulated flight data...")
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
