# %%
PATH="airfares-enhanced.csv"
import pandas as pd
df=pd.read_csv(PATH)
df.head(5)

# Imports
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import sbx
from sbx import PPO, TD3, SAC
from scipy.special import expit
from sbx.common.vec_env import DummyVecEnv, SubprocVecEnv
from sbx.common.env_util import make_vec_env

# %%
# Data Preprocessing
df = pd.read_csv(PATH)
df.fillna(method='ffill', inplace=True)

# Categorical Encoding
categorical_cols = ['flight','airline','source_city','departure_time','stops',
                   'arrival_time','destination_city','class','seasonality']
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Handle dates
df['travel_date']=pd.to_datetime(df['travel_date'])
period = 365.25
df['day_sin'] = np.sin(df['travel_date'].dt.day * (2 * np.pi / period))
df['day_cos'] = np.cos(df['travel_date'].dt.day * (2 * np.pi / period))
df['day_of_month'] = df['travel_date'].dt.month
df['week_of_year'] = df['travel_date'].dt.isocalendar().week
df=df.drop("travel_date", axis=1)

target = 'price'

# Normalize features
scaler = MinMaxScaler()
num_features = ['duration','demand_index','competitor_price',
               'seats_left','adjusted_price']
df[num_features] = scaler.fit_transform(df[num_features])

# For the environment, save min and max price for scaling actions
min_price = df[target].min()
max_price = df[target].max()

# %%
for col in df.columns:
    print(col, df[col].dtype)
    
# %%
# Create Airfare Environment Class
class AirfarePricingEnv(gym.Env):
    def __init__(self, data, min_price, max_price):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_idx = 0
        self.min_price = min_price
        self.max_price = max_price
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # normalized price
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(data.shape[1] - 1,), dtype=np.float32
        )
        # Example coefficients for the demand model (customize as needed)
        self.beta = np.random.uniform(-1, 1, size=(data.shape[1] - 1))
        self.gamma = -2.0  # price sensitivity

    def reset(self):
        self.current_idx = 0
        return self.data.iloc[self.current_idx, :-1].values.astype(np.float32)

    def step(self, action):
        price = float(action[0]) * (self.max_price - self.min_price) + self.min_price
        X = self.data.iloc[self.current_idx, :-1].values

        # Simulate purchase probability (logistic demand model)
        prob_purchase = expit(np.dot(self.beta, X) + self.gamma * price)
        reward = price * prob_purchase
        self.current_idx += 1
        done = self.current_idx >= len(self.data)
        next_state = (
            self.data.iloc[self.current_idx, :-1].values.astype(np.float32)
            if not done
            else np.zeros_like(X, dtype=np.float32)
        )
        return next_state, reward, done, {}

# %%
# Create Metrics Class
class Metrics:
    def __init__(self, model_name):
        self.rewards = []
        self.model_name = model_name
        self.metrics_dir = f'metrics/{model_name}'
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def log_reward(self, reward):
        self.rewards.append(reward)
    
    def save_and_plot(self):
        rewards = np.array(self.rewards)
        np.save(f'{self.metrics_dir}/rewards.npy', rewards)
        window = min(100, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg)
        plt.title(f'Moving Average Reward - {self.model_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig(f'{self.metrics_dir}/reward_curve.png')
        plt.close()

# %%
# Early stopping callback
class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -np.inf
        self.counter = 0
        self.should_stop = False
        self.rewards_history = []
        
    def __call__(self, locals_dict, globals_dict):
        # Get the most recent episode reward
        if 'ep_info_buffer' in locals_dict and locals_dict['ep_info_buffer']:
            latest_reward = locals_dict['ep_info_buffer'][-1]['r']
            self.rewards_history.append(latest_reward)
            
            # Only check after a certain number of episodes
            if len(self.rewards_history) % 10 == 0:
                # Calculate average of last 10 rewards
                avg_reward = np.mean(self.rewards_history[-10:])
                
                # Check for improvement
                if avg_reward > self.best_reward + self.min_delta:
                    self.best_reward = avg_reward
                    self.counter = 0
                else:
                    self.counter += 1
                    
                # Check if we should stop
                if self.counter >= self.patience:
                    self.should_stop = True
                    print(f"Early stopping triggered! No improvement for {self.patience} checks.")
                    return False  # Stop training
                    
        return True  # Continue training

# %%
# Create training function
def train_and_evaluate(model_class, model_name, env_class, env_kwargs, n_envs=4, total_timesteps=100_000):
    print(f"Training {model_name}...")
    
    # Create early stopping callback
    early_stopping = EarlyStoppingCallback(patience=10, min_delta=0.05)
    
    # Create vectorized environment for training
    # PPO benefits most from vectorization
    if model_name == 'ppo':
        n_envs_to_use = n_envs
    else:
        # Off-policy algorithms like TD3 and SAC can work well with fewer envs
        n_envs_to_use = 1
    
    # Create the vectorized environment
    vec_env = make_vec_env(
        env_class, 
        n_envs=n_envs_to_use,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_envs_to_use > 1 else DummyVecEnv
    )
    
    # Different parameters for different algorithms
    if model_name == 'ppo':
        model = model_class(
            'MlpPolicy', 
            vec_env, 
            verbose=1,
            n_steps=512,
            batch_size=64,
            n_epochs=10
        )
    elif model_name == 'td3':
        model = model_class(
            'MlpPolicy', 
            vec_env, 
            verbose=1,
            buffer_size=10000,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1
        )
    elif model_name == 'sac':
        model = model_class(
            'MlpPolicy', 
            vec_env, 
            verbose=1,
            buffer_size=10000,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1
        )
    
    # Train with early stopping
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True,
        callback=early_stopping
    )
    
    model.save(f'models/{model_name}_airfare')
    
    print(f"Evaluating {model_name}...")
    metrics = Metrics(model_name)
    
    # Create a single environment for evaluation (no need for vectorization during eval)
    eval_env = env_class(**env_kwargs)
    obs = eval_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward
        metrics.log_reward(reward)
    
    # Clean up
    vec_env.close()
    
    print(f"{model_name} total evaluation reward: {total_reward}")
    metrics.save_and_plot()
    print(f"{model_name} metrics saved in metrics/{model_name}/")
    return metrics

# %%
# Execute
os.makedirs('models', exist_ok=True)

# Prepare environment arguments
env_kwargs = {
    'data': df,
    'min_price': min_price,
    'max_price': max_price
}

# Train and evaluate each model
model_classes = {'ppo': PPO, 'td3': TD3, 'sac': SAC}
all_metrics = {}

for name, cls in model_classes.items():
    metrics = train_and_evaluate(
        cls, 
        name, 
        AirfarePricingEnv, 
        env_kwargs,
        n_envs=8 if name == 'ppo' else 1  # PPO benefits more from parallelization
    )
    all_metrics[name] = metrics

# Optionally, plot all moving averages for comparison
plt.figure(figsize=(10, 6))
for name, metrics in all_metrics.items():
    rewards = np.array(metrics.rewards)
    window = min(100, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.plot(moving_avg, label=name.upper())
plt.title('Moving Average Reward Comparison')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.tight_layout()
plt.savefig('metrics/reward_comparison.png')
plt.close()
print("All done! Metrics and models saved.")