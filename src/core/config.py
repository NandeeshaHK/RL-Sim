"""Configuration management for the RL application."""

from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """Manager for loading and merging configuration files."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the config manager.
        
        Args:
            config_path: Path to YAML config file. Uses default if None.
        """
        self.config: dict[str, Any] = {}
        
        # Load default config
        default_path = Path(__file__).parents[2] / "config" / "default.yaml"
        if default_path.exists():
            self.config = self._load_yaml(default_path)
        
        # Override with custom config
        if config_path is not None:
            custom = self._load_yaml(Path(config_path))
            self.config = self._deep_merge(self.config, custom)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load YAML file.
        
        Args:
            path: Path to YAML file.
            
        Returns:
            Parsed configuration dictionary.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary.
            override: Dictionary with override values.
            
        Returns:
            Merged dictionary.
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Dot-separated key path (e.g., "training.batch_size").
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        parts = key.split(".")
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-separated key path.
            value: Value to set.
        """
        parts = key.split(".")
        target = self.config
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value

    def save(self, path: str | Path) -> None:
        """Save current configuration to YAML file.
        
        Args:
            path: Output path.
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    @property
    def algorithm(self) -> str:
        """Get selected algorithm name."""
        return self.get("algorithm", "double_dqn")

    @property
    def training(self) -> dict[str, Any]:
        """Get training configuration."""
        return self.get("training", {})

    @property
    def gui(self) -> dict[str, Any]:
        """Get GUI configuration."""
        return self.get("gui", {})

    @property
    def cuda(self) -> dict[str, Any]:
        """Get CUDA configuration."""
        return self.get("cuda", {})
