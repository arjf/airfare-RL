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
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import jax
import jax.numpy as jnp
from functools import partial
import sbx
from sbx import PPO, TD3, SAC
from scipy.special import expit

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

# GPU check
print(f"JAX devices: {jax.devices()}")
if any(dev.platform == 'gpu' for dev in jax.devices()):
    print("Using GPU acceleration")
else:
    print("No GPU found, using CPU")

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
@partial(jax.jit, static_argnums=(0,))
def compute_purchase_prob(beta, gamma, state, price):
    """JAX accelerated purchase probability computation"""
    logit = jnp.dot(beta, state) + gamma * price
    return jax.nn.sigmoid(logit)

@partial(jax.jit, static_argnums=(1,))
def compute_reward(beta, gamma, state, price):
    """JAX accelerated reward computation"""
    prob = compute_purchase_prob(beta, gamma, state, price)
    return price * prob, prob

# %%
# Create environment instances for vectorized environments
def make_env_fn(env_class, **kwargs):
    """Function to create a single environment instance"""
    def _init():
        return env_class(**kwargs)
    return _init

# Create vectorized environment using Gymnasium's built-in vector environments
def make_vec_env(env_class, n_envs=1, env_kwargs=None, use_async=False):
    """Create a properly vectorized environment using Gymnasium's vector API"""
    env_kwargs = {} if env_kwargs is None else env_kwargs
    env_fns = [make_env_fn(env_class, **env_kwargs) for _ in range(n_envs)]
    
    if use_async and n_envs > 1:
        return AsyncVectorEnv(env_fns)  # Parallel execution across processes
    else:
        return SyncVectorEnv(env_fns)   # Sequential execution in a single process

# %%
# Create Airfare Environment Class
class AirfarePricingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data, min_price, max_price, render_mode=None):
        super().__init__()
        # Pre-process data for faster access
        self.data = data.reset_index(drop=True)
        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.data_length = len(self.data)
        self.render_mode = render_mode
        
        self.current_idx = 0
        self.min_price = min_price
        self.max_price = max_price
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # normalized price
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32
        )
        
        # Initialize demand model parameters (now as numpy arrays for JAX)
        self.beta = np.random.uniform(-1, 1, size=self.features.shape[1])
        self.gamma = -2.0  # price sensitivity
        self.episode_reward = 0
        self.prices = []
        self.purchase_probs = []
        
    def reset(self, *, seed=None, options=None):
        # Gymnasium reset requires seed parameter and returns (obs, info)
        super().reset(seed=seed)
        
        self.current_idx = 0
        self.episode_reward = 0
        self.prices = []
        self.purchase_probs = []
        
        # Return observation and info dict
        return self.features[self.current_idx].copy(), {}

    def step(self, action):
        # Convert normalized price to actual price
        price = float(action[0]) * (self.max_price - self.min_price) + self.min_price
        
        # Get current state
        X = self.features[self.current_idx]
        
        reward, prob_purchase = compute_reward(self.beta, self.gamma, X, price)
        
        # Convert to NP
        reward = float(reward)
        prob_purchase = float(prob_purchase)
        
        # Storess
        self.prices.append(price)
        self.purchase_probs.append(prob_purchase)
        self.episode_reward += reward
        
        # Update state
        self.current_idx += 1
        terminated = self.current_idx >= self.data_length
        truncated = False  # We don't truncate episodes early
        
        # Create info dict
        info = {
            'price': price,
            'purchase_prob': prob_purchase,
            'episode_reward': self.episode_reward if terminated else None
        }
        
        # Get next state
        if not terminated:
            next_state = self.features[self.current_idx].copy()
        else:
            next_state = np.zeros_like(X, dtype=np.float32)
        
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            # Simple text rendering of current state
            if self.current_idx > 0:
                price = self.prices[-1]
                prob = self.purchase_probs[-1]
                print(f"Step {self.current_idx}: Price ${price:.2f}, Purchase prob: {prob:.4f}")
        
    def close(self):
        pass

# %%
# Create Metrics Class
class Metrics:
    def __init__(self, model_name):
        self.rewards = []
        self.prices = []
        self.purchase_probs = []
        self.actual_purchases = []
        self.episode_returns = []
        self.step_count = 0
        self.episode_count = 0
        self.model_name = model_name
        self.metrics_dir = f'metrics/{model_name}'
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def log_step(self, reward, price, purchase_prob, purchased=None):
        """Log metrics for a single pricing decision"""
        self.rewards.append(reward)
        self.prices.append(price)
        self.purchase_probs.append(purchase_prob)
        if purchased is not None:
            self.actual_purchases.append(purchased)
        self.step_count += 1
    
    def log_episode(self, episode_return):
        """Log metrics for a complete episode"""
        self.episode_returns.append(episode_return)
        self.episode_count += 1
    
    def save_and_plot(self):
        """Save metrics data and create comprehensive visualization dashboard"""
        # Create directory for plots
        plots_dir = f'{self.metrics_dir}/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save raw data
        np.save(f'{self.metrics_dir}/rewards.npy', np.array(self.rewards))
        np.save(f'{self.metrics_dir}/prices.npy', np.array(self.prices))
        np.save(f'{self.metrics_dir}/purchase_probs.npy', np.array(self.purchase_probs))
        np.save(f'{self.metrics_dir}/episode_returns.npy', np.array(self.episode_returns))
        
        # Create and save plots
        self._plot_rewards(plots_dir)
        self._plot_price_distribution(plots_dir)
        self._plot_purchase_distribution(plots_dir)
        self._plot_price_reward_relationship(plots_dir)
        self._create_business_metrics_summary(plots_dir)
        self._create_dashboard(plots_dir)
    
    def _plot_rewards(self, plots_dir):
        """Plot reward-related metrics"""
        plt.figure(figsize=(12, 6))
        
        # Individual rewards
        plt.subplot(1, 2, 1)
        window = min(100, len(self.rewards))
        if window > 0:
            moving_avg = np.convolve(self.rewards, np.ones(window) / window, mode='valid')
            plt.plot(moving_avg)
            plt.title(f'Moving Average Reward (window={window})')
            plt.xlabel('Decision Step')
            plt.ylabel('Average Reward')
        
        # Episode returns
        plt.subplot(1, 2, 2)
        if len(self.episode_returns) > 0:
            plt.plot(self.episode_returns)
            plt.title('Episode Returns')
            plt.xlabel('Episode')
            plt.ylabel('Total Return')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/rewards.png')
        plt.close()
    
    def _plot_price_distribution(self, plots_dir):
        """Plot price distribution and trends"""
        if len(self.prices) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Price histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.prices, bins=30, alpha=0.7)
        plt.axvline(np.mean(self.prices), color='r', linestyle='dashed', linewidth=1)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # Price over time
        plt.subplot(1, 2, 2)
        window = min(50, len(self.prices))
        if window > 0:
            moving_avg = np.convolve(self.prices, np.ones(window) / window, mode='valid')
            plt.plot(moving_avg)
            plt.title('Price Trend Over Time')
            plt.xlabel('Decision Step')
            plt.ylabel('Average Price')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/price_distribution.png')
        plt.close()
    
    def _plot_purchase_distribution(self, plots_dir):
        """Plot purchase probability distribution"""
        if len(self.purchase_probs) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Purchase probability histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.purchase_probs, bins=30, alpha=0.7)
        plt.axvline(np.mean(self.purchase_probs), color='r', linestyle='dashed', linewidth=1)
        plt.title('Purchase Probability Distribution')
        plt.xlabel('Purchase Probability')
        plt.ylabel('Frequency')
        
        # Purchase rate over time
        plt.subplot(1, 2, 2)
        window = min(50, len(self.purchase_probs))
        if window > 0:
            moving_avg = np.convolve(self.purchase_probs, np.ones(window) / window, mode='valid')
            plt.plot(moving_avg)
            plt.title('Purchase Rate Trend')
            plt.xlabel('Decision Step')
            plt.ylabel('Average Purchase Probability')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/purchase_distribution.png')
        plt.close()
    
    def _plot_price_reward_relationship(self, plots_dir):
        """Plot relationship between price and reward"""
        if len(self.prices) == 0 or len(self.rewards) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Price vs Reward scatter
        plt.subplot(1, 2, 1)
        plt.scatter(self.prices, self.rewards, alpha=0.3, s=10)
        plt.title('Price vs Reward')
        plt.xlabel('Price')
        plt.ylabel('Reward')
        
        # Group prices and calculate average reward per price bin
        if len(self.prices) > 100:  # Only if we have enough data
            bins = np.linspace(min(self.prices), max(self.prices), 20)
            indices = np.digitize(self.prices, bins)
            avg_rewards = [np.mean([r for p, r in zip(self.prices, self.rewards) if p >= bins[i-1] and p < bins[i]]) 
                          for i in range(1, len(bins))]
            
            plt.subplot(1, 2, 2)
            plt.plot(bins[:-1] + (bins[1] - bins[0])/2, avg_rewards, 'ro-')
            plt.title('Average Reward by Price')
            plt.xlabel('Price')
            plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/price_reward_relationship.png')
        plt.close()
    
    def _create_business_metrics_summary(self, plots_dir):
        """Create business metrics summary"""
        if len(self.prices) == 0 or len(self.rewards) == 0:
            return
            
        # Calculate business metrics
        avg_reward = np.mean(self.rewards)
        avg_price = np.mean(self.prices)
        avg_purchase_prob = np.mean(self.purchase_probs)
        estimated_revenue = np.sum(self.rewards)
        revenue_per_offer = estimated_revenue / len(self.rewards) if len(self.rewards) > 0 else 0
        
        # Price points analysis
        optimal_price_idx = np.argmax([np.mean([r for p, r in zip(self.prices, self.rewards) 
                                               if abs(p - price_point) < (max(self.prices) - min(self.prices))/20])
                                      for price_point in np.linspace(min(self.prices), max(self.prices), 20)])
        optimal_price = np.linspace(min(self.prices), max(self.prices), 20)[optimal_price_idx]
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.95, f'Business Metrics Summary - {self.model_name}', 
                 horizontalalignment='center', fontsize=16, fontweight='bold')
        
        plt.text(0.1, 0.85, f'Total Pricing Decisions: {len(self.rewards)}', fontsize=12)
        plt.text(0.1, 0.80, f'Total Episodes: {len(self.episode_returns)}', fontsize=12)
        plt.text(0.1, 0.75, f'Average Reward: {avg_reward:.4f}', fontsize=12)
        plt.text(0.1, 0.70, f'Average Price: {avg_price:.2f}', fontsize=12)
        plt.text(0.1, 0.65, f'Average Purchase Probability: {avg_purchase_prob:.4f}', fontsize=12)
        plt.text(0.1, 0.60, f'Estimated Total Revenue: {estimated_revenue:.2f}', fontsize=12)
        plt.text(0.1, 0.55, f'Revenue per Offer: {revenue_per_offer:.4f}', fontsize=12)
        plt.text(0.1, 0.50, f'Estimated Optimal Price Point: {optimal_price:.2f}', fontsize=12)
        plt.text(0.1, 0.45, f'Price Range: {min(self.prices):.2f} - {max(self.prices):.2f}', fontsize=12)
        
        plt.axis('off')
        plt.savefig(f'{plots_dir}/business_metrics_summary.png')
        plt.close()
    
    def _create_dashboard(self, plots_dir):
        """Create a combined dashboard of all metrics"""
        plt.figure(figsize=(20, 20))
        
        # Load all previously saved plots if they exist
        plot_files = [
            ('rewards.png', 'Reward Metrics'),
            ('price_distribution.png', 'Price Distribution'),
            ('purchase_distribution.png', 'Purchase Probability'),
            ('price_reward_relationship.png', 'Price-Reward Relationship'),
            ('business_metrics_summary.png', 'Business Metrics')
        ]
        
        for i, (filename, title) in enumerate(plot_files):
            if os.path.exists(f'{plots_dir}/{filename}'):
                img = plt.imread(f'{plots_dir}/{filename}')
                plt.subplot(3, 2, i+1)
                plt.imshow(img)
                plt.title(title)
                plt.axis('off')
        
        plt.suptitle(f'Pricing Model Dashboard - {self.model_name}', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'{self.metrics_dir}/dashboard.png')
        plt.close()

# %%
# Early stopping callback
class EarlyStoppingCallback:
    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -float('inf')
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
                avg_reward = jnp.mean(jnp.array(self.rewards_history[-10:]))
                
                # Check for improvement
                if avg_reward > self.best_reward + self.min_delta:
                    self.best_reward = float(avg_reward)
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
    
    # Determine number of environments based on algorithm
    if model_name == 'ppo':
        # PPO benefits from more parallelization
        n_envs_to_use = n_envs
        use_async = True  # Use async for PPO (more efficient for on-policy)
    else:
        # Off-policy algorithms don't need as many parallel envs
        n_envs_to_use = max(1, n_envs // 2)
        use_async = False  # No need for async for off-policy algorithms
    
    print(f"Using {n_envs_to_use} parallel environments for {model_name}")
    
    # Create a vectorized environment
    if n_envs_to_use > 1:
        vec_env = make_vec_env(env_class, n_envs=n_envs_to_use, env_kwargs=env_kwargs, use_async=use_async)
        env = vec_env
    else:
        # Just use a single environment if only one is needed
        env = env_class(**env_kwargs)
    
    # Configure models with optimized parameters for each algorithm
    try:
        if model_name == 'ppo':
            model = model_class(
                'MlpPolicy', 
                env, 
                verbose=1,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                ent_coef=0.01,  # Encourage exploration
                use_sde=True,   # Stochastic exploration
                sde_sample_freq=4
            )
        elif model_name == 'td3':
            model = model_class(
                'MlpPolicy', 
                env, 
                verbose=1,
                buffer_size=100000,
                learning_starts=1000,
                train_freq=1,
                gradient_steps=1,
                learning_rate=3e-4,
                policy_delay=2,
                batch_size=256
            )
        elif model_name == 'sac':
            model = model_class(
                'MlpPolicy', 
                env, 
                verbose=1,
                buffer_size=100000,
                learning_starts=1000,
                train_freq=1,
                gradient_steps=1,
                learning_rate=3e-4,
                batch_size=256,
                ent_coef='auto'  # Automatic entropy tuning
            )
    except ValueError as e:
        print(f"Error creating {model_name} model: {e}")
        try:
            # Fallback to a single environment if vectorized environment causes issues
            print("Falling back to single environment")
            if isinstance(env, (SyncVectorEnv, AsyncVectorEnv)):
                env.close()
            env = env_class(**env_kwargs)
            if model_name == 'ppo':
                model = model_class('MlpPolicy', env, verbose=1)
            elif model_name == 'td3':
                model = model_class('MlpPolicy', env, verbose=1)
            elif model_name == 'sac':
                model = model_class('MlpPolicy', env, verbose=1)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise
    
    # Enable SBX's JIT compilation if available
    if hasattr(model, 'policy') and hasattr(model.policy, 'set_jit'):
        print(f"Enabling JIT compilation for {model_name}")
        model.policy.set_jit(True)
    
    # Train with early stopping
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True,
        callback=early_stopping
    )
    
    # Save model
    model.save(f'models/{model_name}_airfare')
    
    print(f"Evaluating {model_name}...")
    metrics = Metrics(model_name)
    
    # Create a single environment for evaluation
    eval_env = env_class(**env_kwargs)
    obs, _ = eval_env.reset()  # Gymnasium reset returns (obs, info)
    terminated = truncated = False
    done = False
    total_reward = 0
    
    # Fast evaluation loop
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Log detailed metrics
        metrics.log_step(
            reward=reward,
            price=info['price'],
            purchase_prob=info['purchase_prob']
        )
        
        # Log episode return when episode is done
        if done and 'episode_reward' in info:
            metrics.log_episode(info['episode_reward'])
    
    # Clean up
    if isinstance(env, (SyncVectorEnv, AsyncVectorEnv)):
        env.close()
    eval_env.close()
    
    print(f"{model_name} total evaluation reward: {total_reward}")
    metrics.save_and_plot()
    print(f"{model_name} metrics saved in metrics/{model_name}/")
    return metrics

# %%
# Execute with optimized settings
def main():
    os.makedirs('models', exist_ok=True)
    
    # Prepare environment arguments
    env_kwargs = {
        'data': df,
        'min_price': min_price,
        'max_price': max_price
    }
    
    # Detect number of CPU cores for parallelization
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
    optimal_envs = max(1, min(16, n_cpu - 1)) 
    print(f"Using {optimal_envs} environments for parallel training (detected {n_cpu} CPUs)")
    
    # Train and evaluate each model
    model_classes = {'ppo': PPO, 'td3': TD3, 'sac': SAC}
    all_metrics = {}
    
    for name, cls in model_classes.items():
        try:
            metrics = train_and_evaluate(
                cls, 
                name, 
                AirfarePricingEnv, 
                env_kwargs,
                n_envs=optimal_envs
            )
            all_metrics[name] = metrics
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    # Plot comparison of all successful models
    if all_metrics:
        plt.figure(figsize=(10, 6))
        for name, metrics in all_metrics.items():
            rewards = np.array(metrics.rewards)
            window = min(100, len(rewards))
            if window > 0:
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

if __name__ == "__main__":
    main()