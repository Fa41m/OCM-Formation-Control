# Swarm Robotics Navigation: Hybrid Optimised Control Motion

> Simulation, Training, Evaluation, and Visualization of Autonomous Robot Swarms

---

## Overview

This project implements a **swarm robotics framework** where multiple agents navigate a path with dynamic obstacle avoidance. Two control strategies are supported:

- **Optimised Control Model (OCM)** with dynamic parameter adaptation.
- **Reinforcement Learning (RL)** via Proximal Policy Optimization (PPO).

Comprehensive **performance evaluation** and **visualizations** are included to compare the two approaches across various swarm sizes and obstacle difficulties.



---

## Features

- Dynamic parameter adaptation based on obstacle proximity and swarm density.
- Multiple obstacle generation modes with varying difficulty.
- RL training using Stable-Baselines3 PPO.
- Particle Swarm Optimization (PSO) based control parameter optimization.
- Automatic logging of costs, rewards, and per-timestep statistics.
- Plot generation for metrics like cost, collisions, alignment, and control effort.
- One-way ANOVA and Tukey HSD post-hoc testing for swarm performance comparisons.
- Offline simulation video generation every N episodes.

---

## Requirements

Install the following packages:

```bash
pip install gymnasium numpy matplotlib scikit-learn seaborn pandas dataframe_image scipy statsmodels stable-baselines3
```

Also ensure you have **ffmpeg** installed for video saving:

- Ubuntu:
  ```bash
  sudo apt-get install ffmpeg
  ```
- MacOS:
  ```bash
  brew install ffmpeg
  ```
- Windows: [Download FFmpeg here](https://ffmpeg.org/download.html)

---

## Project Structure

| File                    | Purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `env.py`                | Custom `SwarmEnv` class (Gymnasium environment).                   |
| `ocm.py`                | Core swarm behaviors, force models, obstacle generation.           |
| `train_and_simulate.py` | RL training and offline simulation video creation.                 |
| `Simulate_data.py`      | Simulate swarm data using optimised control parameters (PSO).      |
| `graph Generator.py`    | Generate training plots, metric comparisons, statistical analysis. |

Generated outputs (data, plots, videos) are saved under the `Data/` and `videos/` directories.

---

## How to Run

### 1. Simulate Swarm with Optimised Control (PSO)

```bash
python Simulate_data.py
```
It will create directories to store both RL and PSO data if the directories dont exist.
This will generate simulation data for multiple obstacle levels and robot counts under PSO-optimised control.
The parameters can be altered to try different swarm sizes and a more exhausitive search can be conducted by changing the PSO paramters.

### 2. Train a Swarm Policy with RL (PPO)

```bash
python train_and_simulate.py
```
This was used to train the PPO policy where to generate the data, the swarm values were manually changed.
Offline simulation videos and episode reward logs are saved during training.
The values can be altered to create a more sophistaced PPO policy and can be trained for a longer period of time.

### 3. Generate Plots and Statistical Analysis

```bash
python graph\ Generator.py
```
This is used to turn the raw CSV data into detailed graphs to help understand how the algorithms work.
Functions are present to store both RL and PSO data in the corresponding folders.
Graphs and summary statistics will be saved in the corresponding `plots_combined/` folders.

---

## Key Concepts

- **Dynamic Parameter Tuning:** `alpha`, `beta`, `K`, and `C` are adjusted based on obstacle proximity and minimum inter-robot distance to ensure safe, efficient motion.
- **Obstacle Generation Levels:** Varying obstacle configurations (offset, in-path, paired passages) increase navigation complexity.
- **ANOVA and Tukey HSD:** Statistical tests used to verify if swarm size significantly impacts performance metrics.
- **Visualization:** Matplotlib and Seaborn plots are automatically generated to analyze and report results.

---

## Future Work

- Integrate albation studies to better understand the effect of each parameter
- Realistic Simulation Environments such as Webots to get more realistic data and perform real world tests
- Enhanced RL Architecture as it currently uses the base SB3 MLPpolicy
- Advanced Perception via Computer Vision as currently it uses LiDAR sensors

---

## License

This project is licensed under the MIT License.

---

