# Mario Reinforcement Learning Agent
**(Double-Dueling-DQN with NoisyNet and N-Step Replay)**

## Overview

This project implements a **Reinforcement Learning (RL) agent** based on advanced **DQN variants** to train a Super Mario Bros player.  
The main techniques used are:

- **Double DQN**: Helps mitigate overestimation of action values.
- **Dueling DQN**: Separates state-value and advantage estimation for better policy learning.
- **Noisy Networks (NoisyNet)**: Replaces ε-greedy exploration with parameterized, learnable noise.
- **N-Step Replay Buffer**: More stable learning by considering multi-step returns.
- **Curriculum Learning**: Structured training phases to prevent premature local optima.

The training follows the spirit of **DQN** and its permitted extensions according to the assignment requirements.

## Key Features

- **Frame Skipping** and **Max-Pooling** to reduce computation and extract motion.
- **Frame Stacking** (4 frames) for temporal information.
- **Randomized No-Op Start** for better generalization.
- **Life-Based Early Episode Termination** to handle Super Mario Bros "death" scenarios.
- **Adaptive Exploration** via Noisy Linear Layers.

## Training Strategy

To avoid the agent falling into local optima early in training, the training is divided into three conceptual phases:

| Phase | Episodes | Purpose |
| :--- | :--- | :--- |
| **Phase 1** | 300 episodes | **Randomized levels** (random 1-1 to 1-4) to encourage broad exploration. |
| **Phase 2** | Planned 500 episodes per level | **Sequential level training** (1-1, 1-2, 1-3, 1-4) — **skipped during actual training**. |
| **Phase 3** | Planned 5000 episodes (full game) | **End-to-end full game training**. |

> **Important:** Although the code plans for about 5000 episodes, in reality, 2000 episodes were sufficient for achieving strong performance, and training was stopped early.

## Agent Architecture

- **Input**: Stack of 4 grayscale frames (each 84×84 pixels).
- **Convolutional Backbone**: Two convolutional layers (DeepMind DQN style).
- **Head**:
  - Shared fully connected NoisyLinear layer.
  - Split into two NoisyLinear branches:
    - One estimating **State Value**.
    - One estimating **Action Advantages**.
- **Output**: Q-values for each action in the `COMPLEX_MOVEMENT` action space.

## Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Batch Size | 512 |
| Replay Buffer Capacity | 200,000 transitions |
| Learning Rate | 1e-4 |
| Discount Factor (Gamma) | 0.99 |
| Target Network Update Frequency | 5,000 steps |
| Max Gradient Norm | 5.0 |
| N-Step Return | 5 agent action blocks (≈ 20 frames) |


## Acknowledgments

- Developed for Assignment 3 — Deep Reinforcement Learning (DQN)  
  National Taiwan University, Spring 2025.
- Techniques inspired by research on Double DQN, Dueling Networks, and NoisyNet.
