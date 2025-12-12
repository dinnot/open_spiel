import os
import torch
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

# --- CONFIGURATION ---
GAME_NAME = "pro_yams"
CHECKPOINT_PATH = "/tmp/pro_yams_dqn_parallel/agent0_step500000.pt" 
HIDDEN_LAYERS = [512, 256, 128]

# Detect device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_agent(env):
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # Recreate the agent
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=HIDDEN_LAYERS,
        replay_buffer_capacity=100,
        batch_size=32,
        loss_str="huber"
    )
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        exit()
        
    print(f"🧠 Loading brain from: {CHECKPOINT_PATH}")
    
    # Load weights
    # map_location ensures it loads correctly even if moving between GPU/CPU machines
    agent._q_network = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Explicitly move to the correct device
    agent._q_network.to(DEVICE)
    agent._q_network.eval() 
    return agent

def main():
    game = pyspiel.load_game(GAME_NAME)
    env = rl_environment.Environment(game)
    
    ai_agent = load_agent(env)
    
    print("\n" + "="*50)
    print(f"🎲  PRO YAMS ARENA (Device: {DEVICE})  🎲")
    print("="*50)
    
    human_pid = 0 
    ai_pid = 1
    
    state = game.new_initial_state()
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            print(f"\n🎲 Dice Roll: {state.action_to_string(action)}")
            state.apply_action(action)
            continue
            
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        
        print(f"\n--- State (Player {current_player}) ---")
        print(str(state)) 
        
        if current_player == human_pid:
            # --- HUMAN TURN ---
            print(f"\n👉 YOUR TURN (Player {human_pid})")
            print("Legal Actions:")
            for i, action in enumerate(legal_actions):
                print(f"  [{i}] {state.action_to_string(action)}")
            
            while True:
                try:
                    choice = input(f"Select Move [0-{len(legal_actions)-1}]: ")
                    idx = int(choice)
                    if 0 <= idx < len(legal_actions):
                        action = legal_actions[idx]
                        break
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Please enter a number.")
            
            state.apply_action(action)
            
        else:
            # --- AI TURN ---
            print(f"\n🤖 AI TURN (Player {ai_pid})")
            
            # 1. Get observation from game
            obs = state.observation_tensor(ai_pid)
            
            # 2. Convert to Tensor and Move to GPU (FIXED)
            obs_tensor = torch.Tensor(np.array([obs])).to(DEVICE)
            
            # 3. Get Q-values
            with torch.no_grad():
                # .cpu() moves result back to CPU so we can use .numpy()
                q_values = ai_agent._q_network(obs_tensor).cpu().numpy()[0]
            
            # 4. Mask illegal moves
            mask = np.ones(ai_agent._num_actions, dtype=bool)
            mask[legal_actions] = False
            q_values[mask] = -float('inf')
            
            # 5. Pick Move
            best_action = np.argmax(q_values)
            action_str = state.action_to_string(best_action)
            print(f"AI chooses: {action_str}")
            print(f"(Confidence: {q_values[best_action]:.4f})")
            
            state.apply_action(best_action)

    print("\n" + "="*50)
    print("🏁 GAME OVER 🏁")
    returns = state.returns()
    print(f"Your Score: {returns[human_pid]}")
    print(f"AI Score:   {returns[ai_pid]}")
    
    if returns[human_pid] > returns[ai_pid]:
        print("🎉 VICTORY!")
    else:
        print("💀 DEFEAT!")
    print("="*50)

if __name__ == "__main__":
    main()