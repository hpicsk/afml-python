"""
A comprehensive, professional-grade backtesting and quantitative finance library.

This library provides a modular toolkit for:
- Running vectorized and event-driven portfolio backtests.
- Detecting common backtesting dangers and assessing robustness.
- Performing efficient, resource-aware analysis on large datasets.
- Optimizing portfolios with various methodologies.
- Analyzing market microstructure features.
- Generating realistic synthetic financial data.
"""

# Core Backtesting Engines
from .core.backtester import VectorizedBacktester
from .core.streaming import StreamingBacktester, KalmanFilterBacktester

# Portfolio Management
from .portfolio.optimizer import PortfolioOptimizer
from .portfolio.analysis import PortfolioAnalytics
from .portfolio import strategies

# Validation and Analysis Tools
from .validation.dangers import DangerDetector
from .validation.robustness import RobustnessChecker
from .validation.efficiency import EfficiencyAnalyzer

# Data Simulation and Handling
from .data.simulation import SyntheticData
from .data.hf_simulation import HighFrequencyDataSimulator
from .data.handler import EfficientDataHandler

# Microstructure Analysis
from .microstructure.analysis import MarketMicrostructureAnalyzer
from .microstructure.processor import TickDataProcessor

# Utility Classes
from .utils.performance import PerformanceMonitor
from .utils.visualization import VisualizationTools

__version__ = "1.0.0"

__all__ = [
    # Core
    "VectorizedBacktester",
    "StreamingBacktester",
    "KalmanFilterBacktester",
    # Portfolio
    "PortfolioOptimizer",
    "PortfolioAnalytics",
    "strategies",
    # Validation
    "DangerDetector",
    "RobustnessChecker",
    "EfficiencyAnalyzer",
    # Data
    "SyntheticData",
    "HighFrequencyDataSimulator",
    "EfficientDataHandler",
    # Microstructure
    "MarketMicrostructureAnalyzer",
    "TickDataProcessor",
    # Utils
    "PerformanceMonitor",
    "VisualizationTools",
] 