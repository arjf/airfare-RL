import numpy as np
import gym
from gym import spaces


class AirlinePricingEnv(gym.Env):
    def __init__(self, data, max_steps=30):
        super(AirlinePricingEnv, self).__init__()

        # Properly assign the data parameter first
        self.data = data
        self.max_steps = max_steps
        self.current_step = 0

        # Now you can safely access data attributes
        self.airlines = self.data["airline"].unique()
        self.days_left_values = sorted(self.data["days_left"].unique(), reverse=True)
        self.min_price = self.data["price"].min() / 100  # Scale down for RL
        self.max_price = self.data["price"].max() / 100

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.5]), high=np.array([2.0]), dtype=np.float32
        )

        # State: [days_to_departure, current_price_ratio, seats_remaining,
        #        airline_code, time_of_day, num_stops, duration]
        self.observation_space = spaces.Box(
            low=np.array([1, 0.5, 0, 0, 0, 0, 1.5]),
            high=np.array([30, 2.0, 180, len(self.airlines), 5, 2, 30]),
            dtype=np.float32,
        )

    def reset(self):
        """Reset environment with a randomly selected flight scenario"""
        # Select random days_left to simulate different booking timeframes
        self.days_to_departure = np.random.choice(self.days_left_values)

        # Get flights with similar days_left
        similar_flights = self.data[self.data["days_left"] == self.days_to_departure]

        # Select a random flight as starting point
        self.current_flight = similar_flights.sample(1).iloc[0]

        # Initialize state variables
        self.airline = self.current_flight["airline"]
        self.airline_code = self.current_flight["airline_code"]
        self.departure_time_num = self.current_flight["departure_time_num"]
        self.stops = self.current_flight["stops_num"]
        self.duration = self.current_flight["duration"]
        self.base_price = self.current_flight["price"]
        self.current_price = self.base_price
        self.seats_remaining = 180
        self.current_step = 0

        return self._get_observation()
