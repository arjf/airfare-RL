import numpy as np
import gym
from gym import spaces


class AirlinePricingEnv(gym.Env):
    """Airline pricing environment for reinforcement learning"""

    def __init__(self, data, max_steps=100):
        super(AirlinePricingEnv, self).__init__()

        # Load and preprocess data
        self.data = data
        self.max_steps = max_steps
        self.current_step = 0

        # Environment parameters
        self.base_demand = 100  # Base demand for the flight
        self.base_price = 200  # Base price in dollars
        self.elasticity = 1.5  # Price elasticity
        self.seats_remaining = 180
        self.days_to_departure = 90

        # Define action and observation space
        # Actions: Set price as percentage of base price (0.5x to 2.0x)
        # Using continuous action space for compatibility with both algorithms
        self.action_space = spaces.Box(
            low=np.array([0.5]), high=np.array([2.0]), dtype=np.float32
        )

        # State space: [days_to_departure, seats_remaining, current_price,
        #               recent_demand, competitor_price]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([90, 180, 500, 100, 500]),
            dtype=np.float32,
        )

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.seats_remaining = 180
        self.days_to_departure = 90
        self.current_price = self.base_price
        self.recent_demand = 0
        self.competitor_price = self.base_price * (0.9 + 0.2 * np.random.random())

        return self._get_observation()

    def _get_observation(self):
        """Return the current state observation"""
        return np.array(
            [
                self.days_to_departure,
                self.seats_remaining,
                self.current_price,
                self.recent_demand,
                self.competitor_price,
            ],
            dtype=np.float32,
        )

    def _calculate_demand(self, price):
        """Calculate customer demand based on price elasticity"""
        # Base demand calculation using price elasticity
        relative_price = price / self.base_price
        price_factor = relative_price ** (-self.elasticity)

        # Adjust for days to departure (demand increases as departure approaches)
        time_factor = 1 + 0.5 * (1 - self.days_to_departure / 90)

        # Adjust for competition
        comp_factor = 1 + 0.3 * (self.competitor_price / price - 1)

        # Calculate final demand
        demand = self.base_demand * price_factor * time_factor * comp_factor

        # Add some randomness
        demand = int(demand * (0.8 + 0.4 * np.random.random()))

        # Ensure demand is non-negative
        return max(0, demand)

    def step(self, action):
        """Execute one step in the environment"""
        # Get price multiplier from continuous action
        price_multiplier = float(action[0])  # Convert to float for safety
        new_price = self.base_price * price_multiplier

        # Calculate demand at this price
        demand = self._calculate_demand(new_price)

        # Cap demand by remaining seats
        bookings = min(demand, self.seats_remaining)

        # Calculate reward (revenue for this step)
        reward = bookings * new_price

        # Update environment state
        self.current_price = new_price
        self.seats_remaining -= bookings
        self.days_to_departure -= 1
        self.recent_demand = demand
        self.current_step += 1

        # Update competitor price
        self.competitor_price = self.competitor_price * (
            0.95 + 0.1 * np.random.random()
        )

        # Check if episode is done
        done = (
            self.current_step >= self.max_steps
            or self.seats_remaining <= 0
            or self.days_to_departure <= 0
        )

        info = {
            "price": new_price,
            "demand": demand,
            "bookings": bookings,
            "revenue": reward,
            "seats_remaining": self.seats_remaining,
            "days_to_departure": self.days_to_departure,
        }

        return self._get_observation(), reward, done, info
