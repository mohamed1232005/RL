import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

# 1. Define the Hanoi Tower Environment
class HanoiTowerEnv(gym.Env):
    def __init__(self, num_disks=3):
        self.num_disks = num_disks
        self.pegs = [list(range(num_disks, 0, -1)), [], []]  # Initial state
        # Action space: 6 actions (from peg i to peg j, i != j)
        # 0: peg 0->1, 1: peg 0->2, 2: peg 1->0, 3: peg 1->2, 4: peg 2->0, 5: peg 2->1
        self.action_space = gym.spaces.Discrete(6)
        # State space: 3^num_disks (each disk can be on any of 3 pegs)
        self.observation_space = gym.spaces.Discrete(3 ** num_disks)

    def reset(self):
        self.pegs = [list(range(self.num_disks, 0, -1)), [], []]
        return self.get_state()

    def get_state(self):
        # Encode state: each disk's position as a base-3 number
        state = 0
        disk_positions = [0] * self.num_disks
        
        for peg_idx in range(3):
            for disk in self.pegs[peg_idx]:
                disk_positions[disk - 1] = peg_idx
        
        for i in range(self.num_disks):
            state += disk_positions[i] * (3 ** i)
        
        return state

    def decode_state(self, state):
        # Convert state number back to peg configuration
        pegs = [[], [], []]
        disk_positions = []
        
        for i in range(self.num_disks):
            peg_idx = (state // (3 ** i)) % 3
            disk_positions.append(peg_idx)
        
        for disk in range(self.num_disks, 0, -1):
            pegs[disk_positions[disk - 1]].append(disk)
        
        return pegs

    def is_valid_move(self, from_peg, to_peg, pegs):
        # Check if move is valid
        if len(pegs[from_peg]) == 0:
            return False
        if len(pegs[to_peg]) > 0 and pegs[from_peg][-1] > pegs[to_peg][-1]:
            return False
        return True

    def step(self, action):
        # Decode action: 0->0to1, 1->0to2, 2->1to0, 3->1to2, 4->2to0, 5->2to1
        action_map = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        from_peg, to_peg = action_map[action]
        
        # Check if move is valid
        if not self.is_valid_move(from_peg, to_peg, self.pegs):
            # Invalid move: large penalty, stay in same state
            return self.get_state(), -10, False, {}
        
        # Execute move
        disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk)
        
        reward = -1  # Cost of each move
        done = self.is_done()
        
        if done:
            reward = 100  # Large reward for solving
        
        return self.get_state(), reward, done, {}

    def is_done(self):
        # Check if all disks are on the third peg
        return self.pegs[2] == list(range(self.num_disks, 0, -1))


# 2. Policy Iteration Algorithm
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=100):
    print(f"\n=== Policy Iteration ===")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of actions: {env.action_space.n}")
    
    # Initialize random policy and value function
    policy = defaultdict(lambda: random.choice(range(env.action_space.n)))
    V = defaultdict(float)
    
    def get_transition(state, action):
        # Simulate taking action from state
        pegs = env.decode_state(state)
        temp_env = HanoiTowerEnv(env.num_disks)
        temp_env.pegs = copy.deepcopy(pegs)
        next_state, reward, done, _ = temp_env.step(action)
        return next_state, reward, done
    
    def policy_evaluation():
        # Evaluate current policy
        iteration = 0
        while True:
            delta = 0
            for state in range(env.observation_space.n):
                v = V[state]
                action = policy[state]
                next_state, reward, done = get_transition(state, action)
                
                if done:
                    V[state] = reward
                else:
                    V[state] = reward + gamma * V[next_state]
                
                delta = max(delta, abs(v - V[state]))
            
            iteration += 1
            if delta < theta:
                print(f"  Policy evaluation converged in {iteration} iterations")
                break
    
    def policy_improvement():
        # Improve policy based on value function
        policy_stable = True
        
        for state in range(env.observation_space.n):
            old_action = policy[state]
            action_values = []
            
            for action in range(env.action_space.n):
                next_state, reward, done = get_transition(state, action)
                
                if done:
                    action_value = reward
                else:
                    action_value = reward + gamma * V[next_state]
                
                action_values.append(action_value)
            
            # Choose best action
            policy[state] = np.argmax(action_values)
            
            if old_action != policy[state]:
                policy_stable = False
        
        return policy_stable
    
    # Main policy iteration loop
    iteration = 0
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}:")
        policy_evaluation()
        
        if policy_improvement():
            print(f"\nPolicy converged after {iteration + 1} iterations!")
            break
        
        iteration += 1
    
    return policy, V


# 3. Test the optimal policy
def test_policy(env, policy, max_steps=50):
    print("\n=== Testing Optimal Policy ===")
    action_names = ["0→1", "0→2", "1→0", "1→2", "2→0", "2→1"]
    
    state = env.reset()
    print(f"Initial state: {env.pegs}")
    
    for step in range(max_steps):
        action = policy[state]
        print(f"\nStep {step + 1}: Action {action_names[action]}")
        
        next_state, reward, done, _ = env.step(action)
        print(f"  Pegs: {env.pegs}")
        print(f"  Reward: {reward}")
        
        state = next_state
        
        if done:
            print(f"\n✓ Solved in {step + 1} steps!")
            return step + 1
    
    print("\n✗ Failed to solve within step limit")
    return -1


# 4. Visualize the solution
def visualize_policy(policy, env):
    print("\n=== Visualization ===")
    action_names = ["Peg 0 → Peg 1", "Peg 0 → Peg 2", "Peg 1 → Peg 0", 
                    "Peg 1 → Peg 2", "Peg 2 → Peg 0", "Peg 2 → Peg 1"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def draw_state(pegs, title=""):
        ax.clear()
        
        # Draw pegs
        for i in range(3):
            ax.plot([i, i], [0, env.num_disks + 1], 'k-', linewidth=3)
            ax.text(i, -0.5, f'Peg {i}', ha='center', fontsize=12, fontweight='bold')
        
        # Draw disks
        for peg_idx, peg in enumerate(pegs):
            for disk_idx, disk in enumerate(peg):
                width = disk * 0.15
                rect = plt.Rectangle((peg_idx - width/2, disk_idx + 0.2), 
                                    width, 0.6, 
                                    color=plt.cm.rainbow(disk/env.num_disks))
                ax.add_patch(rect)
                ax.text(peg_idx, disk_idx + 0.5, str(disk), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.8, 2.8)
        ax.set_ylim(-1, env.num_disks + 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Execute policy and visualize
    state = env.reset()
    draw_state(env.pegs, "Initial State")
    plt.pause(1.5)
    
    steps = 0
    max_steps = 50
    
    while steps < max_steps:
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        
        steps += 1
        draw_state(env.pegs, f"Step {steps}: {action_names[action]}")
        plt.pause(1.0)
        
        state = next_state
        
        if done:
            draw_state(env.pegs, f"✓ Solved in {steps} steps!")
            plt.pause(2.0)
            break
    
    plt.show()


# Main execution
if __name__ == "__main__":
    # Create environment
    num_disks = 3
    env = HanoiTowerEnv(num_disks=num_disks)
    
    print(f"Hanoi Tower Problem with {num_disks} disks")
    print(f"Number of possible states: {env.observation_space.n}")
    print(f"Number of possible actions: {env.action_space.n}")
    
    # Apply policy iteration
    optimal_policy, value_function = policy_iteration(env, gamma=0.99, theta=1e-6)
    
    # Test the policy
    env.reset()
    steps = test_policy(env, optimal_policy)
    
    # Visualize the solution
    env.reset()
    visualize_policy(optimal_policy, env)