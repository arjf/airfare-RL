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
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load data from Kaggle
    print("Loading flight data from Kaggle...")
    data = load_flight_data()
    
    # Display dataset information
    print("\nDataset Information:")
    print(f"Number of records: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    print("\nSample data:")
    print(data.head())
    
    # Create environment
    env = AirlinePricingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Train SAC agent
    print("\nTraining SAC agent...")
    sac_training_metrics = train_agent('SAC', data, episodes=500, save_dir='models')
    
    # Train PPO agent
    print("\nTraining PPO agent...")
    ppo_training_metrics = train_agent('PPO', data, episodes=500, save_dir='models')
    
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

def load_flight_data():
    """
    Download and load airline pricing data from Kaggle
    
    Dataset: "Airline Pricing and Customer Response Dataset"
    This dataset contains:
    - Historical booking patterns
    - Pricing information
    - Competitor pricing
    - Customer demand data
    - Seasonal factors
    """
    import os
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    import pandas as pd
    import zipfile
    
    # Set up the data directory
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    dataset_path = os.path.join(data_dir, 'airline_pricing.csv')
    
    # Check if dataset already exists locally
    if not os.path.exists(dataset_path):
        print("Downloading dataset from Kaggle...")
        
        # Initialize Kaggle API
        # Note: You need to have a kaggle.json file in ~/.kaggle/
        try:
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset (replace with actual dataset name)
            # Format: username/dataset-name
            api.dataset_download_files(
                'airlineindustry/airline-pricing-and-demand-dataset',
                path=data_dir
            )
            
            # Extract the downloaded zip file
            zip_path = os.path.join(data_dir, 'airline-pricing-and-demand-dataset.zip')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                
            # Remove the zip file after extraction
            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Falling back to simulated data...")
            return simulate_flight_data()
    else:
        print("Using existing dataset from local storage.")
    
    # Load and preprocess the dataset
    try:
        # Read the CSV file
        df = pd.read_csv(dataset_path)
        
        # Perform necessary preprocessing
        
        # 1. Handle missing values
        df.fillna({
            'base_demand': df['base_demand'].mean(),
            'base_price': df['base_price'].mean(),
            'elasticity': df['elasticity'].mean(),
            'capacity': df['capacity'].median()
        }, inplace=True)
        
        # 2. Filter out outliers
        df = df[(df['base_price'] > 0) & (df['base_price'] < 2000)]
        df = df[(df['elasticity'] > 0.1) & (df['elasticity'] < 5.0)]
        
        # 3. Feature engineering - add any additional features needed
        # Example: Calculate price per seat
        if 'capacity' in df.columns:
            df['price_per_seat'] = df['base_price'] / df['capacity']
        
        print(f"Successfully loaded dataset with {len(df)} records.")
        return df
        
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        print("Falling back to simulated data...")
        return simulate_flight_data()


def simulate_flight_data():
    """Generate simulated flight data for demonstration"""
    print("Generating simulated flight data...")
    np.random.seed(42)
    n_flights = 100
    
    data = {
        'flight_id': range(1, n_flights + 1),
        'base_demand': np.random.normal(100, 20, n_flights),
        'base_price': np.random.normal(200, 30, n_flights),
        'elasticity': np.random.uniform(1.0, 2.0, n_flights),
        'capacity': np.random.choice([150, 180, 220], n_flights)
    }
    
    return pd.DataFrame(data)



if __name__ == "__main__":
    main()
