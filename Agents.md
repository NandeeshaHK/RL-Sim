# Agents.md - Project Navigation Guide

Quick reference for navigating and extending the Rainbow Lunar Lander project. For newer feartures and updates, keep this document updated.

---

## 🗂️ Directory Structure

```
rainbow_lunarlander/
├── main.py                 ← Entry point, run with: uv run python main.py
├── config/default.yaml     ← Hyperparameters (learning rate, gamma, epsilon, etc.)
├── src/
│   ├── algorithms/         ← RL algorithms (Double DQN, Rainbow)
│   ├── buffers/            ← Experience replay implementations
│   ├── environments/       ← Gymnasium wrappers
│   ├── gui/                ← PyQt6 interface
│   ├── core/               ← Training loop, config, metrics
│   └── utils/              ← CUDA manager, helpers
└── tests/                  ← Unit tests
```

---

## 🧪 Common Experiments

### 1. Tune Hyperparameters
**File:** `config/default.yaml`
```yaml
training:
  learning_rate: 0.0001  # Try: 0.00005 - 0.001
  gamma: 0.99            # Try: 0.95 - 0.999
  batch_size: 64         # Try: 32, 64, 128
```

### 2. Modify Epsilon Decay
**File:** `src/algorithms/double_dqn.py` → `decay_epsilon()`
```python
def decay_epsilon(self) -> None:
    self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)
```

### 3. Change Network Architecture
**File:** `src/algorithms/networks/dqn_networks.py`
- Modify `hidden_dims` in `DQNNetwork` or `DuelingDQNNetwork`

### 4. Adjust Rainbow Components
**File:** `src/algorithms/rainbow_dqn.py`
- `n_atoms`: Distribution resolution (default: 51)
- `n_steps`: Multi-step return length (default: 3)
- Noisy std in `DistributionalDuelingNetwork`

---

## 🔧 Common Upgrades

### Add a New Algorithm

1. Create `src/algorithms/your_algorithm.py`
2. Inherit from `BaseAgent`
3. Implement required methods:
   - `select_action()` - Action selection
   - `train_step()` - Training logic
   - `store_transition()` - Buffer storage
   - `get_q_values()` - For visualization

4. Register in `src/algorithms/__init__.py`
5. Add to GUI dropdown in `src/gui/panels/control_panel.py`

### Add a New Graph

**File:** `src/gui/panels/metrics_panel.py`
```python
# In _setup_ui(), add:
self.new_graph = GraphWidget(title="My Metric", x_label="step", y_label="value")
new_tab = QWidget()
new_layout = QVBoxLayout(new_tab)
new_layout.addWidget(self.new_graph)
self.tab_widget.addTab(new_tab, "My Tab")
```

### Change Environment

**File:** `src/core/trainer.py` → `setup()`
```python
self.env = gym.make("YourEnv-v1", render_mode="rgb_array")
self.state_dim = <your_state_dim>
self.action_dim = <your_action_dim>
```

---

## 📁 Key Files Reference

| Task | File |
|------|------|
| **Run app** | `main.py` |
| **Config** | `config/default.yaml` |
| **Double DQN** | `src/algorithms/double_dqn.py` |
| **Rainbow DQN** | `src/algorithms/rainbow_dqn.py` |
| **Replay Buffer** | `src/buffers/replay_buffer.py` |
| **PER Buffer** | `src/buffers/prioritized_buffer.py` |
| **Neural Networks** | `src/algorithms/networks/dqn_networks.py` |
| **Training Loop** | `src/core/trainer.py` |
| **Main Window** | `src/gui/main_window.py` |
| **Env Panel** | `src/gui/panels/environment_panel.py` |
| **Graphs** | `src/gui/panels/metrics_panel.py` |
| **CUDA Manager** | `src/utils/cuda_manager.py` |

---

## ⚡ Quick Commands

```bash
# Run GUI
uv run python main.py

# Run with Rainbow
uv run python main.py --algorithm rainbow

# Run tests
uv run pytest tests/ -v

# CPU only
uv run python main.py --no-cuda
```

---

## 🔧 Troubleshooting

### CUDA / PyTorch Issues
If `uv run` keeps reinstalling the CPU version of PyTorch:
1. Ensure `pyproject.toml` has the CUDA index configured:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-cu124"
   url = "https://download.pytorch.org/whl/cu124"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-cu124" }
   ```
2. Run `uv sync` to update the lockfile.
3. Verify with:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 🎯 Performance Tips

1. **Faster training:** Increase speed to "Max" in GUI
2. **VRAM issues:** Reduce `batch_size` in config
3. **Compare algorithms:** Run Double DQN first, then switch to Rainbow
4. **Save progress:** Use Save button before closing
