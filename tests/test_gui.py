"""Tests for GUI components."""

import pytest

# Note: These tests require a display or virtual framebuffer
# Skip if no display available

pytest.importorskip("PyQt6")


def test_imports() -> None:
    """Test that all GUI modules can be imported."""
    from src.gui.main_window import MainWindow
    from src.gui.panels.environment_panel import EnvironmentPanel
    from src.gui.panels.network_panel import NetworkPanel
    from src.gui.panels.metrics_panel import MetricsPanel
    from src.gui.panels.control_panel import ControlPanel
    from src.gui.widgets.graph_widget import GraphWidget
    
    # Just verify imports work
    assert MainWindow is not None
    assert EnvironmentPanel is not None
    assert NetworkPanel is not None
    assert MetricsPanel is not None
    assert ControlPanel is not None
    assert GraphWidget is not None
