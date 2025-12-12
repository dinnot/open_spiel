import os
import torch
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.pytorch import dqn

# --- Configuration ---
GAME_NAME = "pro_yams"
NUM_TRAIN_EPISODES = 200_000
NUM_ENVS = 20  # Number of parallel games (adjust based on your CPU cores, 32 is good for Ryzen 9)
EVAL_EVERY = 20
SAVE_EVERY = 10_000
OUTPUT_DIR = "/tmp/pro_yams_dqn"

# Ensure reproducibility and device setup
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

# Create the vectorized environment
# This creates NUM_ENVS independent games running in parallel
def make_env():
    return rl_environment.Environment(GAME_NAME)

envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
info_state_size = envs.observation_spec()["info_state"][0]
num_actions = envs.envs[0].action_spec()["num_actions"]

print(f"Game: {GAME_NAME}")
print(f"Parallel Environments: {NUM_ENVS}")
print(f"State Size: {info_state_size}")
print(f"Num Actions: {num_actions}")

# Define the agents
hidden_layers = [128, 128]

agents = [
    dqn.DQN(
        player_id=idx,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers,
        replay_buffer_capacity=50_000,
        batch_size=128,
        learning_rate=0.001,
        learn_every=10,
        epsilon_decay_duration=50_000,
        epsilon_start=1.0,
        epsilon_end=0.1,
    )
    for idx in range(2)
]

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Training Loop ---
# Initial reset
time_steps = envs.reset()

for episode in range(0, NUM_TRAIN_EPISODES, NUM_ENVS):
    # Check if games are done
    # Note: VectorEnv automatically resets finished environments, but we need to track it
    
    # Collect actions from agents for all environments
    # time_steps is a list of N TimeStep objects
    actions_p0 = []
    actions_p1 = []
    
    # We must process each environment individually to decide who acts
    # (Since games are independent, env 1 might be P0's turn while env 2 is P1's turn)
    step_outputs = []
    
    for i, ts in enumerate(time_steps):
        if ts.last():
            # Game ended in the previous step. The env is already reset.
            # The agents need to learn from the final transition
            for agent in agents:
                agent.step(ts)
        
        # Get action for the current state (or new state if just reset)
        current_player = ts.observations["current_player"]
        
        # In simultaneous games, this logic would differ, but Pro Yams is sequential.
        if current_player == pyspiel.PlayerId.TERMINAL:
            # Should not happen often with auto-reset, but handle just in case
            step_outputs.append(agents[0].step(ts)) # Dummy step
        else:
            step_outputs.append(agents[current_player].step(ts))

    # Apply actions to all environments at once
    # VectorEnv expects a list of StepOutputs (which contain the action)
    time_steps, rewards, dones, _ = envs.step(step_outputs, reset_if_done=True)

    # --- Evaluation ---
    if (episode + 1) % EVAL_EVERY < NUM_ENVS:
        # Simple evaluation using the first environment
        # (For a proper eval, you'd pause training and run a separate loop, 
        # but this is a quick check using the live envs)
        total_reward = 0
        if dones[0]:
            print(f"Episode {episode}: Completed a batch of {NUM_ENVS} games.")

    # --- Checkpointing ---
    if (episode + 1) % SAVE_EVERY < NUM_ENVS:
        for agent in agents:
            agent.save(OUTPUT_DIR)
        print(f"Saved agents to {OUTPUT_DIR}")

print("Training complete!")
