import os
import cv2
import torch
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np

# === Import your updated Noisy Dueling DQN and wrappers ===
from old_noisy import DuelingDQN, ResizeObservation, ObservationStack, RandomStartEnv, RepeatAndMaxEnv

# === Updated environment without early reset on life loss ===
def make_env_full_game():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = RandomStartEnv(env)
    env = RepeatAndMaxEnv(env, repeat=4)
    env = ResizeObservation(env)
    env = ObservationStack(env, k=4)
    return env

def preprocess(frame, normalize=True):
    arr = np.array(frame).transpose(2, 0, 1)
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env_full_game()

    # Load model
    model = DuelingDQN(4, env.action_space.n).to(device)
    model.load_state_dict(torch.load("mario_q_noisy.pth", map_location=device))
    model.eval()

    # Initial observation
    obs = env.reset()
    state = preprocess(obs, normalize=True)

    # Setup video writer
    first_frame = env.render(mode="rgb_array")
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(
        "mario_noisy_full_game.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (width, height)
    )

    # Run game
    total_reward = 0.0
    done = False

    while not done:
        with torch.no_grad():
            action = int(model(state).argmax(1).item())

        obs, reward, done, info = env.step(action)
        state = preprocess(obs, normalize=True)
        total_reward += reward

        frame = env.render(mode="rgb_array")
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Game over! Total reward across all lives: {total_reward:.2f}")

    # Cleanup
    out.release()
    env.close()

if __name__ == "__main__":
    main()
