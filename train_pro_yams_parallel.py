import os
import time
import torch
import pyspiel
import numpy as np
import multiprocessing as mp
import re
from open_spiel.python import rl_environment
from open_spiel.python import rl_agent
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.pytorch import dqn

# --- Configuration ---
GAME_NAME = "pro_yams"
NUM_TRAIN_STEPS = 500_000 
NUM_ENVS = 70             # 70 Processes
LEARN_EVERY = 4           
EVAL_EVERY = 1000         
SAVE_EVERY = 10_000
OUTPUT_DIR = "/tmp/pro_yams_dqn_parallel"
HIDDEN_LAYERS = [512, 256, 128]
EPSILON_DECAY = 5_000_000 

# --- Multiprocessing Worker ---
def worker(rank, pipe, game_name):
    env = rl_environment.Environment(game_name)
    while True:
        try:
            cmd, data = pipe.recv()
            if cmd == 'reset': pipe.send(env.reset())
            elif cmd == 'step': pipe.send(env.step(data))
            elif cmd == 'close': break
        except EOFError: break

class ProcessVectorEnv:
    def __init__(self, num_envs, game_name):
        self.num_envs = num_envs
        self.pipes = []
        self.procs = []
        for rank in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(target=worker, args=(rank, child_conn, game_name))
            proc.daemon = True
            proc.start()
            self.pipes.append(parent_conn)
            self.procs.append(proc)
            
        dummy_env = rl_environment.Environment(game_name)
        self.observation_spec = dummy_env.observation_spec
        self.action_spec = dummy_env.action_spec
        self.num_actions = dummy_env.action_spec()["num_actions"]

    def reset(self):
        for pipe in self.pipes: pipe.send(('reset', None))
        return [pipe.recv() for pipe in self.pipes]

    def step(self, step_outputs, reset_if_done=True):
        for i, pipe in enumerate(self.pipes):
            pipe.send(('step', [step_outputs[i].action]))

        next_time_steps = [pipe.recv() for pipe in self.pipes]
        rewards = []
        dones = []
        unreset_time_steps = [None] * self.num_envs
        
        for i in range(self.num_envs):
            ts = next_time_steps[i]
            rewards.append(ts.rewards if ts.rewards else [0.0, 0.0])
            dones.append(ts.last())
            if ts.last():
                unreset_time_steps[i] = ts 
                if reset_if_done:
                    self.pipes[i].send(('reset', None))
                    next_time_steps[i] = self.pipes[i].recv()
            
        return next_time_steps, rewards, dones, unreset_time_steps

    def close(self):
        for pipe in self.pipes: pipe.send(('close', None))
        for proc in self.procs: proc.join()

# --- Helpers ---
class RandomBot:
    def __init__(self, player_id, num_actions):
        self.player_id = player_id
        self.num_actions = num_actions  
    def step(self, time_step, is_evaluation=True):
        legal = time_step.observations["legal_actions"][self.player_id]
        action = np.random.choice(legal) if legal else 0
        return rl_agent.StepOutput(action=action, probs=[])

def get_actions_batch(agent, info_states, legal_actions_lists, device, is_evaluation=False):
    obs_tensor = torch.Tensor(np.array(info_states)).to(device)
    with torch.no_grad():
        q_values = agent._q_network(obs_tensor).cpu().numpy()
        
    epsilon = agent._get_epsilon(is_evaluation)
    actions = []
    
    for i, legal_actions in enumerate(legal_actions_lists):
        if not legal_actions:
            actions.append(0)
            continue
        if not is_evaluation and np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
        else:
            masked_q = q_values[i].copy()
            mask = np.ones(agent._num_actions, dtype=bool)
            mask[legal_actions] = False
            masked_q[mask] = -1e18
            action = np.argmax(masked_q)
        actions.append(action)
    return actions

def load_checkpoint(agents, output_dir):
    if not os.path.exists(output_dir): return 0
    files = os.listdir(output_dir)
    steps = []
    for f in files:
        match = re.match(r"agent0_step(\d+).pt", f)
        if match: steps.append(int(match.group(1)))
    
    if not steps: return 0
    latest_step = max(steps)
    print(f"🔄 Found checkpoint at Step {latest_step}. Resuming...")

    for idx, agent in enumerate(agents):
        path = os.path.join(output_dir, f"agent{idx}_step{latest_step}.pt")
        if os.path.exists(path):
            agent.load(path)
            agent._step_counter = latest_step * NUM_ENVS
    return latest_step

def main():
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device} with {NUM_ENVS} parallel PROCESSES")

    # --- Setup ---
    envs = ProcessVectorEnv(NUM_ENVS, GAME_NAME)
    info_state_size = envs.observation_spec()["info_state"][0]
    num_actions = envs.num_actions

    print(f"Game: {GAME_NAME}")
    print(f"State Size: {info_state_size}")
    print(f"Num Actions: {num_actions}")

    agents = [
        dqn.DQN(
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=HIDDEN_LAYERS,
            replay_buffer_capacity=200_000,
            batch_size=512,
            learning_rate=0.00005, 
            learn_every=1,
            epsilon_decay_duration=EPSILON_DECAY,
            epsilon_start=1.0,
            epsilon_end=0.05,
            loss_str="huber"
        )
        for idx in range(2)
    ]

    for agent in agents:
        agent._q_network.to(device)
        agent._target_q_network.to(device)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    start_step = load_checkpoint(agents, OUTPUT_DIR)

    # --- PENDING TRANSITIONS BUFFER ---
    # Stores: [state_obs, action, accumulated_reward] for each env/player
    # We wait until the player sees the board AGAIN before learning.
    pending_transitions = [[None, None] for _ in range(NUM_ENVS)]
    
    recent_scores = []
    last_loss = 0.0
    time_steps = envs.reset()
    last_time = time.time()

    print(f"🚀 Starting training from step {start_step}...")

    try:
        for step in range(start_step + 1, NUM_TRAIN_STEPS + 1):
            
            p0_env_indices = [i for i, ts in enumerate(time_steps) if ts.observations["current_player"] == 0]
            p1_env_indices = [i for i, ts in enumerate(time_steps) if ts.observations["current_player"] == 1]
            
            step_outputs = [None] * NUM_ENVS
            
            # --- AGENT P0 DECISIONS ---
            if p0_env_indices:
                p0_obs = [time_steps[i].observations["info_state"][0] for i in p0_env_indices]
                p0_legal = [time_steps[i].observations["legal_actions"][0] for i in p0_env_indices]
                p0_acts = get_actions_batch(agents[0], p0_obs, p0_legal, device)
                
                for k, env_idx in enumerate(p0_env_indices):
                    act = p0_acts[k]
                    step_outputs[env_idx] = rl_agent.StepOutput(action=act, probs=[])
                    
                    # 1. Close the loop for pending transition (Previous P0 -> Current P0)
                    if pending_transitions[env_idx][0] is not None:
                        old_obs, old_act, acc_reward = pending_transitions[env_idx][0]
                        
                        # Correct Mask: We are looking at P0's turn, so P0 has legal actions!
                        legal_mask = np.zeros(num_actions)
                        legal_mask[p0_legal[k]] = 1.0
                        
                        scaled_reward = np.clip(acc_reward / 5000.0, -1.0, 1.0)
                        
                        agents[0]._replay_buffer.add(dqn.Transition(
                            info_state=old_obs,
                            action=old_act,
                            reward=scaled_reward,
                            next_info_state=p0_obs[k], # Valid Next State (My Turn)
                            is_final_step=0.0,
                            legal_actions_mask=legal_mask
                        ))
                    
                    # 2. Start new pending transition
                    pending_transitions[env_idx][0] = [p0_obs[k], act, 0.0] 

            # --- AGENT P1 DECISIONS ---
            if p1_env_indices:
                p1_obs = [time_steps[i].observations["info_state"][1] for i in p1_env_indices]
                p1_legal = [time_steps[i].observations["legal_actions"][1] for i in p1_env_indices]
                p1_acts = get_actions_batch(agents[1], p1_obs, p1_legal, device)
                
                for k, env_idx in enumerate(p1_env_indices):
                    act = p1_acts[k]
                    step_outputs[env_idx] = rl_agent.StepOutput(action=act, probs=[])
                    
                    if pending_transitions[env_idx][1] is not None:
                        old_obs, old_act, acc_reward = pending_transitions[env_idx][1]
                        
                        legal_mask = np.zeros(num_actions)
                        legal_mask[p1_legal[k]] = 1.0
                        
                        scaled_reward = np.clip(acc_reward / 5000.0, -1.0, 1.0)
                        
                        agents[1]._replay_buffer.add(dqn.Transition(
                            info_state=old_obs,
                            action=old_act,
                            reward=scaled_reward,
                            next_info_state=p1_obs[k],
                            is_final_step=0.0,
                            legal_actions_mask=legal_mask
                        ))
                        
                    pending_transitions[env_idx][1] = [p1_obs[k], act, 0.0]

            for i in range(NUM_ENVS):
                if step_outputs[i] is None:
                    step_outputs[i] = rl_agent.StepOutput(action=0, probs=[])

            # --- STEP ---
            next_time_steps, rewards, dones, unreset_time_steps = envs.step(step_outputs, reset_if_done=True)

            # --- ACCUMULATE & TERMINATE ---
            for i in range(NUM_ENVS):
                # 1. Accumulate Rewards
                # If P0 has a pending move, add any reward P0 got this step
                if pending_transitions[i][0] is not None:
                    r = rewards[i][0] if rewards[i] else 0.0
                    if not dones[i]:
                        pending_transitions[i][0][2] += r

                if pending_transitions[i][1] is not None:
                    r = rewards[i][1] if rewards[i] else 0.0
                    if not dones[i]:
                        pending_transitions[i][1][2] += r

                # 2. Handle Terminal State (Game Over)
                if dones[i]:
                    final_r_p0 = unreset_time_steps[i].rewards[0] if unreset_time_steps[i].rewards else 0.0
                    final_r_p1 = unreset_time_steps[i].rewards[1] if unreset_time_steps[i].rewards else 0.0
                    
                    recent_scores.append(final_r_p0)

                    # Finalize P0
                    if pending_transitions[i][0] is not None:
                        old_obs, old_act, acc_reward = pending_transitions[i][0]
                        total_r = acc_reward + final_r_p0
                        scaled_reward = np.clip(total_r / 5000.0, -1.0, 1.0)
                        
                        # Terminal Transition (legal_mask = 0 is correct here because it's terminal)
                        agents[0]._replay_buffer.add(dqn.Transition(
                            info_state=old_obs,
                            action=old_act,
                            reward=scaled_reward,
                            next_info_state=unreset_time_steps[i].observations["info_state"][0], 
                            is_final_step=1.0,
                            legal_actions_mask=np.zeros(num_actions)
                        ))
                        pending_transitions[i][0] = None

                    # Finalize P1
                    if pending_transitions[i][1] is not None:
                        old_obs, old_act, acc_reward = pending_transitions[i][1]
                        total_r = acc_reward + final_r_p1
                        scaled_reward = np.clip(total_r / 5000.0, -1.0, 1.0)
                        
                        agents[1]._replay_buffer.add(dqn.Transition(
                            info_state=old_obs,
                            action=old_act,
                            reward=scaled_reward,
                            next_info_state=unreset_time_steps[i].observations["info_state"][1],
                            is_final_step=1.0,
                            legal_actions_mask=np.zeros(num_actions)
                        ))
                        pending_transitions[i][1] = None

            # --- TRAIN ---
            if step > 200: 
                agents[0]._step_counter += NUM_ENVS
                agents[1]._step_counter += NUM_ENVS
                if step % LEARN_EVERY == 0:
                    l0 = agents[0].learn()
                    if l0 is not None: last_loss = l0.item() if hasattr(l0, 'item') else l0
                    agents[1].learn()

            # --- LOG ---
            if step % EVAL_EVERY == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                sps = (EVAL_EVERY * NUM_ENVS) / elapsed if elapsed > 0 else 0
                mean_sp = np.mean(recent_scores) if recent_scores else 0.0
                
                # VS RANDOM
                eval_env = rl_environment.Environment(GAME_NAME)
                random_bot = RandomBot(1, num_actions)
                total_eval_score = 0
                for _ in range(20):
                    ts = eval_env.reset()
                    while not ts.last():
                        pid = ts.observations["current_player"]
                        if pid == 0:
                            obs = ts.observations["info_state"][0]
                            legal = ts.observations["legal_actions"][0]
                            with torch.no_grad():
                                t_obs = torch.Tensor(np.array([obs])).to(device)
                                q = agents[0]._q_network(t_obs).cpu().numpy()[0]
                            mask = np.ones(num_actions, dtype=bool)
                            mask[legal] = False
                            q[mask] = -1e18
                            action = np.argmax(q)
                            step_res = rl_agent.StepOutput(action=action, probs=[])
                        else:
                            step_res = random_bot.step(ts)
                        ts = eval_env.step([step_res.action])
                    total_eval_score += ts.rewards[0]
                
                avg_vs_random = total_eval_score / 20
                eps = agents[0]._get_epsilon(False)
                
                print(f"Step {step}/{NUM_TRAIN_STEPS} | {elapsed:.1f}s ({sps:.0f} steps/s) | Eps: {eps:.3f} | Loss: {last_loss:.5f} | SP Avg: {mean_sp:.1f} | VS RANDOM: {avg_vs_random:.1f}")
                recent_scores = []
                last_time = time.time()

            # --- SAVE ---
            if step % SAVE_EVERY == 0:
                for idx, agent in enumerate(agents):
                    agent.save(os.path.join(OUTPUT_DIR, f"agent{idx}_step{step}.pt"))
                print(f"Saved checkpoint to {OUTPUT_DIR}")

            time_steps = next_time_steps

    finally:
        envs.close()
        print("Parallel Training Complete!")

if __name__ == "__main__":
    main()