# RAINBOW LUNAR LANDER 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, high-performance reinforcement learning experimentation platform featuring Real-Time Visualization for the Lunar Lander environment. Built with **PyTorch**, **PyQt6**, and **Gymnasium**, optimized for CUDA acceleration.

**Keywords:** Reinforcement Learning, RL, Lunar Lander, Rainbow DQN, Double DQN, PyTorch, CUDA, PyQt6, Real-time Visualization, Neural Network Visualization, Gymnasium, AI, Artificial Intelligence, Deep Learning.

---

## 🎥 Demo

<div align="center">
  <video src="https://github.com/NandeeshaHK/RL-Sim/raw/refs/heads/main/assets/public/sample_video.mp4" 
         autoplay 
         loop 
         muted 
         playsinline 
         width="100%">
  </video>
</div>

---

## 🏷️ GitHub Topics (Manual Setup)

Since GitHub topics cannot be set via git commands, please **manually add these topics** to the repository's "About" settings (top-right of repo page):

`reinforcement-learning` `rainbow-dqn` `pytorch` `cuda-optimized` `real-time-visualization` `gymnasium` `lunar-lander` `deep-learning` `ai` `python` `pyqt6` `gpu-acceleration` `custom-metrics`

---

## ✨ Features

- **Advanced RL Algorithms**: 
  - **Double DQN**: Deep Q-Network with double Q-learning and epsilon-greedy exploration.
  - **Rainbow DQN**: State-of-the-art integration of C51, PER, Dueling Nets, Noisy Nets, and N-step returns.
- **Real-time Visualization**: 
  - 🖥️ **Live Rendering**: Watch the agent learn in real-time.
  - 🧠 **Neural Network Viz**: See activations and Q-values propagate through the network.
  - 📈 **Live Metrics**: Monitor Rewards, Loss, Q-values, and V-values with dynamic graphs.
- **Interactive Control Panel**: 
  - Play/Pause/Stop training.
  - Variable speed control (1x to Max).
  - Hot-swappable algorithms.
- **High Performance**: 
  - Multi-instance support (train multiple agents).
  - CUDA optimization for 4GB VRAM budgets.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/NandeeshaHK/RL-Sim.git
cd RL-Sim

# Install with UV (Recommended for speed & CUDA support)
uv sync

# Verify CUDA installation
uv run check.torch.py
```

## 🚀 Quick Start

```bash
# Launch the GUI
uv run python main.py

# Launch with specific algorithm
uv run python main.py --algorithm rainbow
```

## 📂 Project Structure

```
rainbow_lunarlander/
├── src/
│   ├── algorithms/      # RL Agents (DQN, Rainbow)
│   ├── buffers/         # Replay Buffers (Uniform, PER)
│   ├── environments/    # Gym Wrappers
│   ├── gui/             # PyQt6 Panels & Widgets
│   ├── core/            # Training Loop & Metrics
│   └── utils/           # CUDA & Helper utils
├── config/              # Hyperparameters (YAML)
├── assets/              # Images, Videos, Styles
└── main.py              # Application Entry Point
```

## ⚙️ Configuration

Customize your experiments in `config/default.yaml`:
- **Hyperparameters**: Learning rate, gamma, batch size.
- **Network**: Hidden layer sizes, atom counts (C51).
- **Environment**: Reward shaping, stacking.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
