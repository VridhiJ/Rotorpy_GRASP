import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.wind.dryden_winds import DrydenGust

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP

baseline_controller = SE3Control(quad_params)

# Set up directories
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies", "PPO")
output_dir = os.path.join(os.path.dirname(__file__), "..", "rotorpy", "data_out", "ppo_hover")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to extract epoch number from filename
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

# User selects two models
print("Select two models to compare:")
models_available = os.listdir(models_dir)
for i, name in enumerate(models_available):
    print(f"{i}: {name}")
model_indices = [int(input("Enter first model index: ")), int(input("Enter second model index: "))]

# Load both models
models = []
for idx in model_indices:
    model_dir = os.path.join(models_dir, models_available[idx])
    num_timesteps_list = [fname for fname in os.listdir(model_dir) if fname.startswith('hover_')]
    num_timesteps_list_sorted = sorted(num_timesteps_list, key=extract_number)
    print(f"Select an epoch for model {models_available[idx]}:")
    for i, name in enumerate(num_timesteps_list_sorted):
        print(f"{i}: {name}")
    epoch_idx = int(input("Enter epoch index: "))
    model_path = os.path.join(model_dir, num_timesteps_list_sorted[epoch_idx])
    model = PPO.load(model_path)
    models.append(model)

# Monte Carlo simulation parameters
num_mc_runs = 100 
total_errors_model1 = []
total_errors_model2 = []
total_errors_baseline = []

def run_episode(env, policy=None, record_trajectory=False, reference_trajectory=None):
    obs, _ = env.reset()
    terminated = False
    errors = []
    trajectory = []  # Store trajectory points

    step_count = 0
    while not terminated:
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            # SE3 Controller computes control
            state = {'x': obs[0:3], 'v': obs[3:6], 'q': obs[6:10], 'w': obs[10:13]}
            flat = {'x': [0,0,0], 'x_dot': [0,0,0], 'x_ddot': [0,0,0], 
                    'x_dddot': [0,0,0], 'yaw': 0, 'yaw_dot': 0, 'yaw_ddot': 0}
            control_dict = baseline_controller.update(0, state, flat)
            cmd_motor_speeds = control_dict['cmd_motor_speeds']
            action = np.interp(cmd_motor_speeds, [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1,1])
        
        obs, _, terminated, _, _ = env.step(action)
        trajectory.append(obs[0:3])  # Store the trajectory

        if reference_trajectory is not None:
            idx = min(len(reference_trajectory) - 1, step_count)  # Prevent out-of-bounds
            pos_error = np.linalg.norm(obs[0:3] - reference_trajectory[idx])
        else:
            pos_error = np.linalg.norm(obs[0:3])

        errors.append(pos_error)
        step_count += 1
        
    if record_trajectory:
        return np.mean(errors), trajectory
    else:
        return np.mean(errors)

for run in range(num_mc_runs):
    seed = run  # Ensure reproducibility
    np.random.seed(seed)
    
    # Create wind profile with the same seed for all models in this run
    wind = DrydenGust(dt=1/100, sig_wind=np.array([75,75,30]), altitude=2.0)
    
    # Create environments with the same wind and seed
    def make_env(wind_profile, selected_scenario = 0):
        return gym.make("Quadrotor-v0",
                        control_mode='cmd_motor_speeds',
                        reward_fn=hover_reward,
                        quad_params=quad_params,
                        max_time=5,
                        wind_profile=wind_profile,
                        sim_rate=100,
                        render_mode=None,
                        selected_scenario = selected_scenario)
    
    env_model1 = make_env(wind)
    env_model2 = make_env(wind)
    env_baseline = make_env(wind)

    # Run SE3 controller first to get trajectory
    error_baseline, se3_trajectory = run_episode(env_baseline, record_trajectory=True)

    # Run episodes for RL models
    #error_model1 = run_episode(env_model1, models[0], reference_trajectory=se3_trajectory)
    #error_model2 = run_episode(env_model2, models[1], reference_trajectory=se3_trajectory)
    error_model1 = run_episode(env_model1, models[0])
    error_model2 = run_episode(env_model2, models[1])
                               
    total_errors_model1.append(error_model1)
    total_errors_model2.append(error_model2)
    total_errors_baseline.append(error_baseline)
    
    print(f"Run {run+1}/{num_mc_runs}: Model1: {error_model1:.2f}, Model2: {error_model2:.2f}, Baseline: {error_baseline:.2f}")

# Calculate statistics
mean_model1, std_model1 = np.mean(total_errors_model1), np.std(total_errors_model1)
mean_model2, std_model2 = np.mean(total_errors_model2), np.std(total_errors_model2)
mean_baseline, std_baseline = np.mean(total_errors_baseline), np.std(total_errors_baseline)

print("\n--- Results ---")
print(f"Model 1 (Trained without wind): {mean_model1:.2f} ± {std_model1:.2f}")
print(f"Model 2 (Trained with wind): {mean_model2:.2f} ± {std_model2:.2f}")
print(f"Baseline (SE3 Controller): {mean_baseline:.2f} ± {std_baseline:.2f}")

# Plotting
plt.figure()
labels = ['Model 1 (No Wind)', 'Model 2 (Wind)', 'Baseline']
means = [mean_model1, mean_model2, mean_baseline]
stds = [std_model1, std_model2, std_baseline]
plt.bar(labels, means, yerr=stds, capsize=10)
plt.ylabel('Mean Tracking Error (m)')
plt.title('Comparison of Tracking Errors with Monte Carlo Simulations')
plt.savefig(os.path.join(output_dir, 'error_comparison.png'))
plt.show()