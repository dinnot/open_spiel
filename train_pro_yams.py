import os
import torch
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

# --- Configuration ---
GAME_NAME = "pro_yams"
NUM_TRAIN_EPISODES = 200_000
EVAL_EVERY = 2_000
SAVE_EVERY = 10_000
OUTPUT_DIR = "/tmp/pro_yams_dqn"

# Ensure reproducibility and device setup
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

# Create the environment
env = rl_environment.Environment(GAME_NAME)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

print(f"Game: {GAME_NAME}")
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
for episode in range(NUM_TRAIN_EPISODES):
    time_step = env.reset()
    
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        
        if env.is_turn_based:
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
        else:
            action_list = [agent.step(time_step).action for agent in agents]
            
        time_step = env.step(action_list)

    # Episode ended
    for agent in agents:
        agent.step(time_step)

    # --- Evaluation ---
    if (episode + 1) % EVAL_EVERY == 0:
        total_returns = np.zeros(2)
        eval_episodes = 20
        for _ in range(eval_episodes):
            eval_step = env.reset()
            while not eval_step.last():
                pid = eval_step.observations["current_player"]
                action = agents[pid].step(eval_step, is_evaluation=True).action
                eval_step = env.step([action])
            total_returns += np.array(eval_step.rewards)
            
        avg_returns = total_returns / eval_episodes
        print(f"Episode {episode + 1}: Avg Return = {avg_returns[0]}")
        
    # --- Checkpointing (FIXED) ---
    if (episode + 1) % SAVE_EVERY == 0:
        for idx, agent in enumerate(agents):
            # FIX: Create a full file path for each agent
            save_path = os.path.join(OUTPUT_DIR, f"agent{idx}_episode{episode+1}.pt")
            agent.save(save_path)
        print(f"Saved agents to {OUTPUT_DIR}")

print("Training complete!")