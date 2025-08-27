#!/usr/bin/env python3
"""
Mario Double‑Dueling‑DQN with
  • NoisyLinear exploration
  • memory‑saving n‑step replay buffer (per agent action block)
  • three‑phase curriculum (random 1‑1…1‑4 → sequential → full game)
  • simple snapshot checkpointing (mario_q_noisy.pth / mario_q_target_noisy.pth)

Modified to align buffer storage and N-step calculation with agent action steps (every 4 raw frames).
"""

# ───────────────────────── Imports ─────────────────────────
import warnings, os, random, time, math, pickle, argparse
from collections import deque
from typing import Deque, Tuple, Optional

warnings.filterwarnings("ignore",
    message=".*out of date.*",
    category=UserWarning,
    module="gym.envs.registration")

import numpy as np, cv2
import gym, gym_super_mario_bros
from gym import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ───────────────────── Hyper‑parameters ─────────────────────
# ---------- curriculum ----------
PHASE1_EPIS       = 300      # random 1‑1…1‑4
EPIS_PER_LEVEL    = 500      # sequential 1‑1,1‑2,1‑3,1‑4
PHASE3_EPIS       = 5_000      # full game
# ---------- learning  -----------
BATCH_SIZE        = 512
BUFFER_CAPACITY   = 200_000 # Capacity in terms of agent action blocks
LEARNING_RATE     = 1e-4
GAMMA             = 0.99
N_STEPS           = 5 # *** N_STEPS now refers to N agent action blocks ***
TARGET_UPDATE_EVERY = 5_000    # gradient steps (each step uses BATCH_SIZE agent action blocks)
MAX_GRAD_NORM     = 5.0
# ---------- misc -----------------
LOG_INTERVAL      = 10         # episodes
RESUME_TRAINING   = True      # default; can override with --resume

POLICY_PATH = "mario_q_noisy_new.pth"
TARGET_PATH = "mario_q_target_noisy_new.pth"
SCORE_PKL   = "score_noisy_new.p"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────── Environment wrappers ───────────────────
# Keeping RandomStartEnv
class RandomStartEnv(gym.Wrapper):
    def __init__(self, env, max_noop=4):
        super().__init__(env); self.max_noop=max_noop
    def reset(self, **kw):
        obs=self.env.reset(**kw)
        # Perform random no-ops at the start
        for _ in range(random.randint(0,self.max_noop)):
            a = random.randint(0,11)
            obs,_,d,_=self.env.step(a)
            if d: # If done during no-ops, reset again
                obs=self.env.reset(**kw)
        return obs

class LifeResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self.lives,self.real_done=0,True
    def step(self,a):
        obs,r,d,info=self.env.step(a); self.real_done=d
        lives=info.get('life',getattr(self.env.unwrapped,'_life',0))
        # Treat losing a life as done, unless it's the final life
        if lives < self.lives and lives > 0:
             d = True
        self.lives=lives; return obs,r,d,info
    def reset(self, **kw):
        # Only call env.reset() if the game is truly over (lost all lives)
        if self.real_done:
            obs = self.env.reset(**kw)
        else:
            # If just lost a life, step with a no-op to continue from the start of the level
            obs, _, _, _ = self.env.step(0)
        self.lives=getattr(self.env.unwrapped,'_life',0)
        return obs

# Removed RepeatAndMaxEnv, ResizeObs, FrameStack
# as their logic is now handled within the training loop's act logic.
def build_env(level: Optional[str]):
    env_id = "SuperMarioBros-v0" if level is None else f"SuperMarioBros-{level}-v0"
    e=gym_super_mario_bros.make(env_id)
    e=JoypadSpace(e,COMPLEX_MOVEMENT)
    # Include RandomStartEnv and LifeResetEnv
    for w in (RandomStartEnv, LifeResetEnv,): e=w(e)
    return e

# ───────────────── Replay Buffer ──────────────────
# ReplayBuffer stores transitions per agent action block
class ReplayBuffer:
    def __init__(self, capacity:int, n_steps_agent_blocks:int, gamma:float, skip_frames:int):
        # Main buffer stores (s_u8_start_nstep, a_nstep, R_nstep_agent_blocks, s_u8_end_nstep, d_nstep) tuples
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.n_steps_agent_blocks = n_steps_agent_blocks
        self.gamma = gamma
        self.skip_frames = skip_frames

        # Temporary buffer to store agent action block transitions
        # before calculating the N-step return for the oldest one.
        # Stores (s_u8_start, a, R_block, s_u8_end, d_block)
        self._temp_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n_steps_agent_blocks)

    # Push receives a single agent action block transition
    def push(self, s_u8_start: np.ndarray, a: int, R_block: float, s_u8_end: np.ndarray, d_block: bool):
        # Store the 1-step agent block transition in the temporary buffer
        self._temp_buffer.append((s_u8_start, a, R_block, s_u8_end, d_block))

        # If the temporary buffer is full (contains N_STEPS_AGENT_BLOCKS transitions),
        # we can calculate the N-step return for the oldest transition.
        if len(self._temp_buffer) == self.n_steps_agent_blocks:
            # Calculate the N-step return for the first transition in _temp_buffer
            R_nstep_agent_blocks, s_u8_end_nstep, done_nstep = 0.0, None, False
            # _temp_buffer stores (s_u8_start_block_i, a_block_i, R_block_i, s_u8_end_block_i, d_block_i)
            for i, (_, _, R_block_i, s_u8_end_block_i, d_block_i) in enumerate(self._temp_buffer):
                 # Discount factor for reward R_block_i is gamma raised to the power of
                 # the total number of raw frames from the start of the N-step sequence
                 # to the start of block_i. This is i * skip_frames.
                 R_nstep_agent_blocks += (self.gamma** (i * self.skip_frames)) * R_block_i
                 s_u8_end_nstep, done_nstep = s_u8_end_block_i, d_block_i

                 # If done occurs in any block within the sequence, the sequence ends there
                 if d_block_i:
                     # The subsequent R_block_i will be 0 if the episode truly ended.
                     # The Q value for the terminal state will be 0.
                     # The accumulation R_nstep_agent_blocks is correct as is.
                     break

            # The state and action for the N-step agent block transition are from the *start* of the sequence
            s_u8_start_nstep, a_nstep, *_ = self._temp_buffer[0]

            # Add the calculated N-step agent block transition to the main buffer
            self.buf.append((s_u8_start_nstep, a_nstep, R_nstep_agent_blocks, s_u8_end_nstep, done_nstep))

            # Remove the oldest transition from the temporary buffer
            self._temp_buffer.popleft()

        # If done occurs in the last pushed block, flush all remaining sequences in _temp_buffer
        # These are partial N-step sequences.
        if d_block:
            while self._temp_buffer:
                # Calculate N-step return for the sequence starting with the oldest transition
                R_nstep_agent_blocks, s_u8_end_nstep, done_nstep = 0.0, None, False
                # Need to iterate only over the *remaining* elements in _temp_buffer
                for i, (_, _, R_block_i, s_u8_end_block_i, d_block_i) in enumerate(self._temp_buffer):
                     R_nstep_agent_blocks += (self.gamma** (i * self.skip_frames)) * R_block_i
                     s_u8_end_nstep, done_nstep = s_u8_end_block_i, d_block_i
                     if d_block_i: # Should be True for the last element if d_block was True
                         break

                s_u8_start_nstep, a_nstep, *_ = self._temp_buffer[0]
                self.buf.append((s_u8_start_nstep, a_nstep, R_nstep_agent_blocks, s_u8_end_nstep, done_nstep))
                self._temp_buffer.popleft()


    def sample(self,k:int):
        if len(self.buf) < k:
             raise IndexError("Not enough samples in buffer")

        b=random.sample(self.buf,k)
        # b is a list of (s_u8_start_nstep, a_nstep, R_nstep_agent_blocks, s_u8_end_nstep, d_nstep) tuples
        s_u8_batch, a_batch, R_batch, ns_u8_batch, d_batch = zip(*b)

        # Convert numpy arrays to tensors
        # States are (Batch, C, H, W)
        s_tensor = torch.tensor(np.stack(s_u8_batch, axis=0), dtype=torch.float32, device=DEVICE) / 255.0
        a_tensor = torch.tensor(np.array(a_batch), dtype=torch.int64, device=DEVICE).unsqueeze(-1)
        R_tensor = torch.tensor(np.array(R_batch), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        ns_tensor = torch.tensor(np.stack(ns_u8_batch, axis=0), dtype=torch.float32, device=DEVICE).unsqueeze(-1) # Should not unsqueeze here
        ns_tensor = torch.tensor(np.stack(ns_u8_batch, axis=0), dtype=torch.float32, device=DEVICE) / 255.0 # Corrected

        d_tensor = torch.tensor(np.array(d_batch), dtype=torch.bool, device=DEVICE).unsqueeze(-1)

        return s_tensor, a_tensor, R_tensor, ns_tensor, d_tensor

    def __len__(self): return len(self.buf)

# ─────────────── Noisy Linear & Network ───────────────
class NoisyLinear(nn.Module):
    def __init__(self,in_f,out_f,sigma=0.5):
        super().__init__(); self.in_f=in_f
        self.mu_w=nn.Parameter(torch.empty(out_f,in_f))
        self.sig_w=nn.Parameter(torch.empty(out_f,in_f))
        self.mu_b=nn.Parameter(torch.empty(out_f))
        self.sig_b=nn.Parameter(torch.empty(out_f))
        self.register_buffer("eps_in",torch.zeros(in_f))
        self.register_buffer("eps_out",torch.zeros(out_f))
        self.reset_parameters(sigma); self.reset_noise()
    def reset_parameters(self,s):
        bound=1/math.sqrt(self.in_f)
        self.mu_w.data.uniform_(-bound,bound)
        self.mu_b.data.uniform_(-bound,bound)
        self.sig_w.data.fill_(s/math.sqrt(self.in_f))
        self.sig_b.data.fill_(s/math.sqrt(self.mu_w.size(0)))
    @staticmethod
    def _noise(size): x=torch.randn(size); return x.sign()*x.abs().sqrt()
    def reset_noise(self):
        self.eps_in.copy_( self._noise(self.in_f))
        self.eps_out.copy_(self._noise(self.mu_w.size(0)))
    def forward(self,x):
        # During evaluation/action selection, noise is used.
        # During training (td_update), noise is not reset, effectively using mu.
        w=self.mu_w + self.sig_w * self.eps_out.unsqueeze(1)*self.eps_in.unsqueeze(0)
        b=self.mu_b + self.sig_b * self.eps_out
        return F.linear(x,w,b)

class DuelingDQN(nn.Module):
    def __init__(self,in_frames:int,n_act:int):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_frames,32,8,4),nn.ReLU(),
            nn.Conv2d(32,64,3,1),nn.ReLU())
        with torch.no_grad():
            dummy=torch.zeros(1,in_frames,84,84)
            conv_out=self.conv(dummy).view(1,-1).size(1)
        self.fc  = NoisyLinear(conv_out,512)
        self.val = NoisyLinear(512,1)
        self.adv = NoisyLinear(512,n_act)
    def forward(self,x):
        # Input x is expected to be 4x84x84 tensor (normalized 0-1)
        x=self.conv(x).view(x.size(0),-1); x=F.relu(self.fc(x))
        v,a=self.val(x),self.adv(x)
        return v + a - a.mean(1,keepdim=True)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m,NoisyLinear): m.reset_noise()


# ──────────── Custom Observation Processing ────────────

# State variables for the custom processing logic, managed per episode
pool_buf: Deque[np.ndarray] = deque(maxlen=2) # Stores raw 240x256x3 frames
stack_buf: Deque[np.ndarray] = deque(maxlen=4) # Stores processed 84x84 uint8 frames

def reset_processing_state():
    """Resets the state variables for the start of a new episode."""
    global pool_buf, stack_buf
    pool_buf.clear()
    stack_buf.clear()

def process_raw_frame(raw_obs: np.ndarray) -> np.ndarray:
    """
    Processes a raw observation (240x256x3) using max-pooling, grayscale, resize, and stacking.
    Updates internal buffers.

    Returns the processed 4x84x84 uint8 state (np.ndarray).
    """
    global pool_buf, stack_buf

    # Add new frame to max-pool buffer
    pool_buf.append(raw_obs)

    # Max-pool last two frames
    frame = np.maximum(pool_buf[-2], pool_buf[-1]) if len(pool_buf) == 2 else raw_obs # Fallback for very first step

    # Grayscale + resize
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    proc = small.astype(np.uint8)

    # Add to frame stack
    stack_buf.append(proc)
    while len(stack_buf) < 4: # Pad if needed (first few steps of episode)
        stack_buf.append(proc)

    # Construct the processed state (shape: 4x84x84)
    state_u8 = np.stack(list(stack_buf), axis=0) # Shape: 4x84x84

    return state_u8

# ──────────── Train‑step helper ────────────
def td_update(net,tgt,buf,opt, skip_frames: int):
    # We need at least BATCH_SIZE agent action blocks in the main buffer
    if len(buf)<BATCH_SIZE: return None

    # Sample returns tensors already on DEVICE and normalized
    # buf.sample returns (s_u8_start_nstep, a_nstep, R_nstep_agent_blocks, s_u8_end_nstep, d_nstep)
    s,a,R_nstep,ns,d_nstep=buf.sample(BATCH_SIZE)

    # Use net for Q values of the taken actions (s)
    q=net(s).gather(1,a)

    # Use target net for target Q values (Double DQN)
    with torch.no_grad():
        # Select best action for the next state (ns) using the online network (net)
        na=net(ns).argmax(1,keepdim=True)
        # Evaluate the selected action using the target network (tgt)
        # The target is R_nstep + gamma_effective^N_STEPS * Q_target(s_nstep, a_selected_by_online)
        # R_nstep is already the accumulated discounted reward over N_STEPS agent blocks.
        # The state ns is the state after N_STEPS agent blocks.
        # The discount factor between the start state (s) and the end state (ns)
        # of the N-step sequence is gamma raised to the power of (N_STEPS * skip_frames).
        gamma_total = GAMMA ** (N_STEPS * skip_frames)
        y = R_nstep + gamma_total * (~d_nstep)* tgt(ns).gather(1,na)

    loss=F.smooth_l1_loss(q,y)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(),MAX_GRAD_NORM); opt.step()
    return float(loss)

# ───────────────────────── Main loop ─────────────────────────
def main(resume: bool):
    net=DuelingDQN(4,len(COMPLEX_MOVEMENT)).to(DEVICE)
    tgt=DuelingDQN(4,len(COMPLEX_MOVEMENT)).to(DEVICE)
    tgt.load_state_dict(net.state_dict())
    opt=optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if resume and os.path.exists(POLICY_PATH):
        net.load_state_dict(torch.load(POLICY_PATH,map_location=DEVICE))
        tgt.load_state_dict(torch.load(TARGET_PATH,map_location=DEVICE))
        print("✔ loaded saved weights.")

    # Buffer uses N_STEPS for agent action blocks
    buf=ReplayBuffer(BUFFER_CAPACITY, N_STEPS, GAMMA, skip_frames=4) # Pass skip_frames to buffer

    # curriculum sequence
    curriculum = []
    curriculum.append((None, PHASE3_EPIS))  # Phase 3 next
    curriculum.append(("random", PHASE1_EPIS))  # Phase 1 last
    curriculum.extend((lvl, EPIS_PER_LEVEL) for lvl in ["1-1", "1-2", "1-3", "1-4"])  # Levels first



    # Initialize counters and history
    ep, step_ct, log_ct = 0, 0, 0 # ep tracks total episodes completed
    reward_sum, loss_sum, t0 = 0.0,0.0,time.perf_counter()
    score_hist=[] # score_hist stores average rewards per LOG_INTERVAL

    # Load score history and set initial ep if resuming
    if resume and os.path.exists(SCORE_PKL):
         try:
             score_hist = pickle.load(open(SCORE_PKL, "rb"))
             print(f"✔ loaded score history with {len(score_hist)} entries.")
             # Calculate starting episode based on history length and log interval
             # ep should be the total number of episodes *completed* before resuming
             ep = len(score_hist) * LOG_INTERVAL
             print(f"Starting training from episode {ep}")
         except Exception as e:
             print(f"Error loading score history: {e}")
             score_hist = []
             ep = 0 # Start from scratch if loading fails

    # --- Revised Curriculum Loop ---
    total_episodes_completed_before_this_phase = 0
    for phase_idx, (phase, n_ep_in_phase) in enumerate(curriculum):
         total_episodes_after_this_phase = total_episodes_completed_before_this_phase + n_ep_in_phase

         if ep >= total_episodes_after_this_phase:
             # This phase is fully completed, skip it
             print(f"Skipping phase {phase} ({n_ep_in_phase} episodes) as training already passed episode {ep}")
             total_episodes_completed_before_this_phase = total_episodes_after_this_phase
             continue # Go to the next phase

         # If we reach here, we are in this phase or an earlier one.
         # We need to run episodes until ep reaches total_episodes_after_this_phase
         print(f"Starting phase: {phase} ({n_ep_in_phase} episodes). Resuming from episode {ep}.")

         # Run episodes for the current phase until the target episode count is reached
         while ep < total_episodes_after_this_phase:
             # Determine level for this episode
             current_level = random.choice(["1-1","1-2","1-3","1-4"]) if phase=="random" else phase
             env=build_env(current_level)

             # Reset observation processing state for the new episode
             reset_processing_state()

             # Get initial observation (raw 240x256x3)
             # RandomStartEnv handles initial no-ops here
             raw_obs = env.reset()

             # Process the initial raw observation to get the first state
             current_state_u8 = process_raw_frame(raw_obs)

             episode_done = False
             stuck, prev_x=0,0

             # Episode loop iterates per agent action block
             while not episode_done:
                 # --- Start of Agent Action Block ---
                 state_u8_block_start = current_state_u8 # State at the beginning of the block

                 # Select action using the network on the state at the block start
                 state_tensor = torch.tensor(state_u8_block_start, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
                 net.reset_noise() # Reset noise for action selection
                 with torch.no_grad():
                     action_block = int(net(state_tensor).argmax(1).item())

                 total_reward_block = 0.0
                 done_block = False # Done flag for this block

                 # Perform 'skip' raw environment steps with the selected action
                 for _ in range(4): # skip = 4
                     raw_obs, raw_r, done, info = env.step(action_block)

                     # Apply stuck penalty
                     cur_x = max(info.get('x_pos', prev_x), prev_x)
                     stuck = stuck + 1 if cur_x - prev_x <= 0 else 0
                     prev_x = cur_x
                     if stuck >= 600: raw_r -= 0.1
                     if stuck >= 1000: raw_r -= 1; done = True # Set done if stuck
                     if info.get("flag_get", False):
                        raw_r += 50


                     total_reward_block += raw_r # Accumulate raw reward (with penalties)

                     # Process the raw observation after the step
                     current_state_u8 = process_raw_frame(raw_obs)

                     # If done occurs during the block, the block ends early
                     if done:
                         done_block = True
                         break # Break the inner skip loop

                 # --- End of Agent Action Block ---
                 state_u8_block_end = current_state_u8 # State at the end of the block

                 # Push the agent action block transition to the buffer
                 # Buffer stores (s_u8_start, a, R_block, s_u8_end, d_block)
                 buf.push(state_u8_block_start, action_block, total_reward_block, state_u8_block_end, done_block)

                 reward_sum += total_reward_block # Sum block rewards for logging

                 # Check if the episode is done (based on the last raw step)
                 if done_block:
                     episode_done = True # Break the outer agent action block loop

                 # Perform training step if main buffer is sufficiently full
                 # This happens independently of episode steps
                 if len(buf)>=2_000: # Start training after 2000 N-step agent block transitions
                     # Pass skip_frames to td_update for correct gamma calculation
                     loss=td_update(net,tgt,buf,opt, skip_frames=4)
                     if loss is not None:
                         loss_sum+=loss
                         step_ct+=1 # step_ct counts gradient steps (each step uses BATCH_SIZE agent blocks)
                         log_ct += 1 # Count steps where loss was calculated

                         # Update target network and save checkpoints
                         if step_ct % TARGET_UPDATE_EVERY==0:
                             print(f"\nStep {step_ct}: Updating target network and saving checkpoints...")
                             tgt.load_state_dict(net.state_dict())
                             torch.save(net.state_dict(),POLICY_PATH)
                             torch.save(tgt.state_dict(),TARGET_PATH)
                             print("Checkpoints saved.")

             env.close() # Close environment at the end of the episode

             # Increment episode counter *after* the episode finishes
             ep+=1

             # Logging and saving score history
             # Log when ep is a multiple of LOG_INTERVAL
             if ep % LOG_INTERVAL == 0:
                 elapsed=time.perf_counter()-t0
                 avg_r,reward_sum=reward_sum/LOG_INTERVAL,0.0
                 # Calculate average loss only over steps where loss was calculated (log_ct)
                 # Avoid division by zero if no training steps occurred in the interval
                 avg_l=loss_sum/max(1,log_ct) if log_ct > 0 else 0.0
                 loss_sum,log_ct=0.0,0 # Reset sums and count for next interval
                 print(f"{DEVICE} | Ep {ep} | AvgR {avg_r:.2f} | "
                       f"AvgL {avg_l:.4f} | Time {elapsed:.1f}s")
                 score_hist.append(avg_r)
                 pickle.dump(score_hist,open(SCORE_PKL,"wb"))
                 t0=time.perf_counter() # Reset timer for next log interval

         # After the while loop for this phase finishes, update the cumulative count
         total_episodes_completed_before_this_phase = total_episodes_after_this_phase
         # The outer loop continues to the next phase automatically

    print("✓ training finished")
    # Final save
    torch.save(net.state_dict(),POLICY_PATH)
    torch.save(tgt.state_dict(),TARGET_PATH)
    pickle.dump(score_hist,open(SCORE_PKL,"wb")) # Save final score history

# ───────────────────────── entry ───────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--resume",action="store_true",help="resume from snapshots")
    args=ap.parse_args()
    main(args.resume or RESUME_TRAINING)