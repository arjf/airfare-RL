import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import AirlinePricingEnv
from sac_agent import SAC
from ppo_agent import PPO


def evaluate_agent(agent_type, agent, env, episodes=100):
    """Evaluate agent performance"""

    eval_metrics = {
        "total_revenues": [],
        "occupancy_rates": [],
        "price_paths": [],
        "demand_paths": [],
        "execution_times": [],
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_revenue = 0
        price_path = []
        demand_path = []

        # Run episode without exploration
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_revenue += reward

            # Record metrics
            price_path.append(env.current_price)
            demand_path.append(env.recent_demand)

        # Calculate occupancy
        occupancy = (180 - env.seats_remaining) / 180

        # Store episode metrics
        eval_metrics["total_revenues"].append(total_revenue)
        eval_metrics["occupancy_rates"].append(occupancy)
        eval_metrics["price_paths"].append(price_path)
        eval_metrics["demand_paths"].append(demand_path)

    # Calculate average metrics
    avg_revenue = np.mean(eval_metrics["total_revenues"])
    avg_occupancy = np.mean(eval_metrics["occupancy_rates"])

    print(f"Evaluation Results for {agent_type}:")
    print(f"Average Revenue: ${avg_revenue:.2f}")
    print(f"Average Occupancy Rate: {avg_occupancy:.2f}")

    return eval_metrics


def compare_algorithms(sac_metrics, ppo_metrics):
    """Compare SAC and PPO performance"""

    sac_revenue = np.mean(sac_metrics["total_revenues"])
    ppo_revenue = np.mean(ppo_metrics["total_revenues"])

    sac_occupancy = np.mean(sac_metrics["occupancy_rates"])
    ppo_occupancy = np.mean(ppo_metrics["occupancy_rates"])

    # Revenue comparison
    revenue_diff = ((sac_revenue / ppo_revenue) - 1) * 100
    if revenue_diff > 0:
        revenue_msg = f"SAC generates {revenue_diff:.2f}% more revenue than PPO"
    else:
        revenue_msg = f"PPO generates {-revenue_diff:.2f}% more revenue than SAC"

    # Occupancy comparison
    occupancy_diff = ((sac_occupancy / ppo_occupancy) - 1) * 100
    if occupancy_diff > 0:
        occupancy_msg = f"SAC achieves {occupancy_diff:.2f}% higher occupancy than PPO"
    else:
        occupancy_msg = f"PPO achieves {-occupancy_diff:.2f}% higher occupancy than SAC"

    print("\nComparison Results:")
    print(f"SAC Revenue: ${sac_revenue:.2f}")
    print(f"PPO Revenue: ${ppo_revenue:.2f}")
    print(revenue_msg)
    print(f"\nSAC Occupancy: {sac_occupancy:.2f}")
    print(f"PPO Occupancy: {ppo_occupancy:.2f}")
    print(occupancy_msg)

    return {
        "sac_revenue": sac_revenue,
        "ppo_revenue": ppo_revenue,
        "sac_occupancy": sac_occupancy,
        "ppo_occupancy": ppo_occupancy,
        "revenue_diff": revenue_diff,
        "occupancy_diff": occupancy_diff,
    }
