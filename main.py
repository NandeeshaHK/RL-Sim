"""Rainbow Lunar Lander - Main Entry Point.

A modular RL experimentation platform with real-time visualization.
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from src.core.config import ConfigManager
from src.core.trainer import Trainer
from src.gui.main_window import MainWindow


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rainbow Lunar Lander - RL Experimentation Platform"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["double_dqn", "rainbow"],
        default=None,
        help="Algorithm to use (overrides config)",
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA (use CPU only)",
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (training only)",
    )
    
    return parser.parse_args()


class Application:
    """Main application class integrating GUI and training."""

    def __init__(self, config: ConfigManager, algorithm: str | None = None) -> None:
        """Initialize the application.
        
        Args:
            config: Configuration manager.
            algorithm: Algorithm override.
        """
        self.config = config
        self.algorithm = algorithm
        
        # Create Qt application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Rainbow Lunar Lander")
        self.app.setApplicationVersion("0.1.0")
        
        # Create main window
        self.window = MainWindow(config.config)
        
        # Create trainer
        self.trainer = Trainer(
            config,
            on_step=self._on_training_step,
            on_episode_end=self._on_episode_end,
        )
        
        # Training timer
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self._training_tick)
        
        # Connect window signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect window signals to handlers."""
        self.window.control_panel.play_clicked.connect(self._on_play)
        self.window.control_panel.pause_clicked.connect(self._on_pause)
        self.window.control_panel.stop_clicked.connect(self._on_stop)
        self.window.control_panel.reset_clicked.connect(self._on_reset)
        self.window.control_panel.speed_changed.connect(self._on_speed_changed)
        self.window.control_panel.algorithm_changed.connect(self._on_algorithm_changed)
        self.window.control_panel.save_clicked.connect(self._on_save)
        self.window.control_panel.load_clicked.connect(self._on_load)

    def _on_play(self) -> None:
        """Handle play button."""
        if self.trainer.agent is None:
            self.trainer.setup(self.algorithm)
        
        self.trainer.start()
        
        # Calculate timer interval based on speed
        speed = self.trainer.speed_multiplier
        if speed == 0:  # Max speed
            interval = 1  # As fast as possible
        else:
            base_interval = 1000 // self.config.get("gui.render_fps", 30)
            interval = max(1, base_interval // speed)
        
        self.training_timer.start(interval)

    def _on_pause(self) -> None:
        """Handle pause button."""
        self.trainer.pause()
        self.training_timer.stop()

    def _on_stop(self) -> None:
        """Handle stop button."""
        self.trainer.stop()
        self.training_timer.stop()

    def _on_reset(self) -> None:
        """Handle reset button."""
        self.trainer.reset()
        self.training_timer.stop()
        self.window.env_panel.clear()
        self.window.metrics_panel.clear_all()

    def _on_speed_changed(self, speed: int) -> None:
        """Handle speed change."""
        self.trainer.set_speed(speed)
        
        # Adjust timer if running
        if self.training_timer.isActive():
            if speed == 0:
                self.training_timer.setInterval(1)
            else:
                base_interval = 1000 // self.config.get("gui.render_fps", 30)
                self.training_timer.setInterval(max(1, base_interval // speed))

    def _on_algorithm_changed(self, algorithm: str) -> None:
        """Handle algorithm change."""
        self.algorithm = algorithm
        # New algorithm takes effect on next setup

    def _on_save(self) -> None:
        """Handle save."""
        path = Path("checkpoints") / f"checkpoint_{self.trainer.current_episode}.pt"
        self.trainer.save_checkpoint(path)

    def _on_load(self) -> None:
        """Handle load."""
        # Would use file dialog in full implementation
        pass

    def _training_tick(self) -> None:
        """Execute training step(s) on timer tick."""
        if not self.trainer.is_running or self.trainer.is_paused:
            return
        
        # Execute multiple steps for faster speeds
        speed = self.trainer.speed_multiplier
        steps_per_tick = 1 if speed == 0 else speed
        
        for _ in range(min(steps_per_tick, 10)):
            if self.trainer.is_running and not self.trainer.is_paused:
                self.trainer.train_step()

    def _on_training_step(self, **kwargs) -> None:
        """Handle training step callback."""
        self.window.update_training_state(**kwargs)

    def _on_episode_end(self, metrics) -> None:
        """Handle episode end callback."""
        self.window.metrics_panel.update_episode_metrics(
            episode=metrics.episode,
            reward=metrics.reward,
            q_mean=metrics.avg_q_value,
            q_max=metrics.max_q_value,
        )

    def run(self) -> int:
        """Run the application.
        
        Returns:
            Exit code.
        """
        self.window.show()
        return self.app.exec()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.training_timer.stop()
        self.trainer.close()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code.
    """
    args = parse_arguments()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Override CUDA if requested
    if args.no_cuda:
        config.set("cuda.enabled", False)
    
    if args.headless:
        # Headless training mode (no GUI)
        print("Headless mode not yet implemented.")
        return 1
    
    # Run GUI application
    app = Application(config, args.algorithm)
    
    try:
        return app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    sys.exit(main())
