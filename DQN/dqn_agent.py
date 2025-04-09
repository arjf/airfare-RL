"""
Deep Q-Network (DQN) for Airfare Price Prediction
=================================================
This script implements a DQN model to predict airfare prices
using flight details such as airline, time, duration, and days before departure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Create directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# 1. Data Loading and Exploration
print("Loading and exploring data...")
df = pd.read_csv('Clean_Dataset.csv')
print(f"Dataset shape: {df.shape}")

# 2. Data Visualization
def create_visualizations(df):
    # Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Distribution of Flight Prices')
    plt.xlabel('Price (INR)')
    plt.ylabel('Frequency')
    plt.savefig('visualizations/price_distribution.png')
    plt.close()
    
    # Price vs Days Left
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='days_left', y='price', data=df)
    plt.title('Price Distribution by Days Left Before Departure')
    plt.xlabel('Days Left')
    plt.ylabel('Price (INR)')
    plt.savefig('visualizations/price_vs_days_left.png')
    plt.close()
    
    # Price by Airline
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='airline', y='price', data=df)
    plt.title('Price Distribution by Airline')
    plt.xlabel('Airline')
    plt.ylabel('Price (INR)')
    plt.xticks(rotation=45)
    plt.savefig('visualizations/price_by_airline.png')
    plt.close()
    
    # Correlation Heatmap for numeric features
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

# Call visualization function
create_visualizations(df)

# 3. Data Preprocessing
print("\nPreprocessing data...")

# Convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['airline', 'departure_time', 'stops', 'arrival_time'])

# Create price brackets for DQN (discretize the continuous price)
df['price_bracket'] = pd.qcut(df['price'], q=10, labels=False)
num_price_brackets = 10

# Plot price brackets distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='price_bracket', data=df)
plt.title('Distribution of Price Brackets')
plt.xlabel('Price Bracket')
plt.ylabel('Count')
plt.savefig('visualizations/price_bracket_distribution.png')
plt.close()

# Prepare features and targets
X = df_encoded.drop(['price', 'price_bracket', 'flight', 'source_city', 'destination_city', 'class'], axis=1)
y = df['price_bracket']  # Using price brackets as the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. DQN Implementation

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Build DQN Model
def build_dqn_model(state_size, action_size):
    model = Sequential([
        Dense(128, input_dim=state_size, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(capacity=2000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_dqn_model(state_size, action_size)
        self.target_model = build_dqn_model(state_size, action_size)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size):
        if self.memory.size() < batch_size:
            return
        
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 5. Training and Evaluation
print("\nTraining DQN model...")

# Initialize agent
state_size = X_train_scaled.shape[1]
action_size = num_price_brackets
agent = DQNAgent(state_size, action_size)

# Training parameters
batch_size = 32
num_episodes = 100
training_history = []

# Training loop
for e in range(num_episodes):
    # Randomly sample episodes from training data
    indices = np.random.choice(len(X_train_scaled), size=100, replace=False)
    
    total_reward = 0
    
    for i, idx in enumerate(indices):
        state = X_train_scaled[idx].reshape(1, -1)
        action = agent.act(state)
        # Reward is negative of the error between predicted and actual price bracket
        true_bracket = y_train.iloc[idx]
        reward = -abs(action - true_bracket)
        total_reward += reward
        
        # Get next state
        next_idx = indices[(i + 1) % len(indices)]
        next_state = X_train_scaled[next_idx].reshape(1, -1)
        done = (i == len(indices) - 1)
        
        # Remember the experience
        agent.remember(state, action, reward, next_state, done)
        
        # Experience replay
        if agent.memory.size() >= batch_size:
            agent.replay(batch_size)
        
        if done:
            agent.update_target_model()
            break
    
    # Track training progress
    training_history.append({
        'episode': e,
        'total_reward': total_reward,
        'epsilon': agent.epsilon
    })
    
    # Print progress
    if (e+1) % 10 == 0:
        print(f"Episode: {e+1}/{num_episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Plot Training History
train_df = pd.DataFrame(training_history)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_df['episode'], train_df['total_reward'])
plt.title('Reward During Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(train_df['episode'], train_df['epsilon'])
plt.title('Exploration Rate (Epsilon) During Training')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.savefig('visualizations/training_history.png')
plt.close()

# 6. Model Evaluation
print("\nEvaluating the model...")

# Make predictions on test data
predictions = []
for i in range(len(X_test_scaled)):
    state = X_test_scaled[i].reshape(1, -1)
    action = agent.act(state)
    predictions.append(action)

# Convert to numpy array
predictions = np.array(predictions)
y_test_np = y_test.values

# Calculate metrics
prediction_accuracy = np.mean(predictions == y_test_np)
mae = mean_absolute_error(y_test_np, predictions)
rmse = np.sqrt(mean_squared_error(y_test_np, predictions))
r2 = r2_score(y_test_np, predictions)

print(f"Prediction Accuracy: {prediction_accuracy:.4f}")
print(f"Mean Absolute Error: {mae:.4f} price brackets")
print(f"Root Mean Squared Error: {rmse:.4f} price brackets")
print(f"R² Score: {r2:.4f}")

# Create confusion matrix
cm = confusion_matrix(y_test_np, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Price Bracket Predictions')
plt.xlabel('Predicted Price Bracket')
plt.ylabel('Actual Price Bracket')
plt.savefig('visualizations/confusion_matrix.png')
plt.close()

# Compare actual vs predicted price brackets
plt.figure(figsize=(12, 6))
plt.scatter(y_test_np, predictions, alpha=0.5)
plt.plot([0, 9], [0, 9], 'r--')  # Perfect prediction line
plt.title('Actual vs Predicted Price Brackets')
plt.xlabel('Actual Price Bracket')
plt.ylabel('Predicted Price Bracket')
plt.savefig('visualizations/prediction_comparison.png')
plt.close()

# Calculate bracket boundaries for reference
price_bracket_labels = pd.qcut(df['price'], q=10)
price_bracket_bounds = price_bracket_labels.cat.categories

# Create a reference table for price brackets
bounds_df = pd.DataFrame({
    'Bracket': range(num_price_brackets),
    'Price Range': [str(bound) for bound in price_bracket_bounds]
})

print("\nPrice bracket reference:")
print(bounds_df)

# Error analysis by feature
test_results = X_test.copy()
test_results['actual_bracket'] = y_test
test_results['predicted_bracket'] = predictions
test_results['error'] = test_results['actual_bracket'] - test_results['predicted_bracket']

# Extract original categorical columns for error analysis
original_test_data = df.iloc[y_test.index]
for col in ['airline', 'departure_time', 'stops']:
    test_results[col] = original_test_data[col].values

# Plot error by airline
plt.figure(figsize=(12, 6))
sns.boxplot(x='airline', y='error', data=test_results)
plt.title('Prediction Error by Airline')
plt.xlabel('Airline')
plt.ylabel('Error (Actual - Predicted Bracket)')
plt.xticks(rotation=45)
plt.savefig('visualizations/error_by_airline.png')
plt.close()

# 7. Generate Summary Report
with open('dqn_airfare_prediction_report.txt', 'w') as f:
    f.write("DQN Airfare Price Prediction - Summary Report\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Dataset Information:\n")
    f.write(f"- Total records: {len(df)}\n")
    f.write(f"- Features used: {', '.join(X.columns)}\n")
    f.write(f"- Price range: ₹{df['price'].min()} to ₹{df['price'].max()}\n\n")
    
    f.write("Model Configuration:\n")
    f.write(f"- Number of price brackets: {num_price_brackets}\n")
    f.write(f"- Training episodes: {num_episodes}\n")
    f.write(f"- Batch size: {batch_size}\n")
    f.write(f"- Discount factor (gamma): {agent.gamma}\n")
    f.write(f"- Final exploration rate (epsilon): {agent.epsilon:.4f}\n\n")
    
    f.write("Performance Metrics:\n")
    f.write(f"- Prediction Accuracy: {prediction_accuracy:.4f}\n")
    f.write(f"- Mean Absolute Error: {mae:.4f} price brackets\n")
    f.write(f"- Root Mean Squared Error: {rmse:.4f} price brackets\n")
    f.write(f"- R² Score: {r2:.4f}\n\n")
    
    f.write("Price Bracket Reference:\n")
    for _, row in bounds_df.iterrows():
        f.write(f"- Bracket {row['Bracket']}: {row['Price Range']}\n")

print("\nDQN Airfare Price Prediction completed successfully!")
print("Check 'dqn_airfare_prediction_report.txt' for a summary report.")
print("Visualizations have been saved to the 'visualizations/' directory.")
