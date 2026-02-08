# Structure:
# State-Action Visualization: Visualize the state of the Hanoi Tower and show the chosen actions at each step.
# Rendering: Use Gymnasium’s rendering capabilities or matplotlib to create visualizations.
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize the Hanoi Tower problem's current state and policy
def visualize_policy(policy, env):
    # Assume we have a way to display the current state (could be a grid or animation)
    # Here we will visualize the policy, showing which action to take in each state
    fig, ax = plt.subplots()
    ax.set_title("Hanoi Tower Policy Visualization")

    for state in range(env.observation_space.n):
        # For simplicity, visualize each action in the policy as a number in a grid or simple output
        ax.text(state, 0, str(policy[state]), fontsize=12, ha='center', va='center')

    plt.show()

# Load the environment and optimal policy (from the first file)
env = gym.make("HanoiTower-v0")  # Replace with the actual environment name
optimal_policy = np.load("optimal_policy.npy")  # Assuming optimal policy is saved from the main file

# Visualize the policy
visualize_policy(optimal_policy, env)
