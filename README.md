# AFML: Advances in Financial Machine Learning - Refactored Library

This repository contains a refactored Python library based on the concepts and code from the book "Advances in Financial Machine Learning" by Marcos López de Prado. The library provides a modular and extensible framework for backtesting, portfolio optimization, and financial data analysis.

All example scripts have been tested and are confirmed to run correctly with the refactored library.

## Installation

To install the package in development mode, run:

```bash
pip install -e .
```

This will install the package in editable mode, allowing you to make changes to the source code and have them immediately reflected without reinstalling.

## Key Features

- **Data Structures**: Advanced financial data structures like tick, volume, and dollar bars.
- **Labeling Techniques**: Sophisticated methods for labeling financial data, such as the Triple Barrier Method.
- **Feature Analysis**: Tools for feature importance analysis, including Mean Decrease Impurity (MDI) and Mean Decrease Accuracy (MDA).
- **Cross-Validation**: Specialized cross-validation techniques for financial time series, like Purged K-Fold CV.
- **Vectorized Backtesting**: Fast and efficient backtesting of trading strategies.
- **Portfolio Optimization**: A suite of tools for constructing optimal portfolios.
- **Performance Analytics**: In-depth analysis of portfolio performance and risk.
- **Model Validation**: Tools for checking the robustness and validity of financial models.

## Example Scripts

This repository includes several example scripts that demonstrate how to use the refactored library, corresponding to chapters in the book.

- `scripts/chap2_bars.py`: Demonstrates the creation of standard and alternative time series bars (e.g., tick, volume, dollar bars).
- `scripts/chap3_labeling.py`: Implements the Triple Barrier Method for labeling financial data for supervised learning.
- `scripts/chap4_sample_weights.py`: Shows how to compute sample weights to account for overlapping observations in financial time series.
- `scripts/chap5_fractional_differentiation.py`: Illustrates the concept of fractional differentiation to achieve stationarity while preserving memory.
- `scripts/chap8_feature_importance.py`: Provides examples of various feature importance techniques like MDI, MDA, and Single Feature Importance.
- `scripts/chap7_cross_validation.py`: Demonstrates the use of Purged K-Fold Cross-Validation to prevent data leakage in backtesting.
- `scripts/chap8_feature_importance_analysis.py`: Expands on feature importance by analyzing clustering and pairwise importance of features.
- `scripts/chap14_backtest_statistics.py`: Shows how to compute various backtest performance statistics, such as Sharpe ratio, drawdown, and precision.
- `scripts/chap10_bet_sizing.py`: Implements different bet sizing techniques based on model predictions and confidence.
- `scripts/chap11_dangers_of_backtesting.py`: Demonstrates how to use the `DangerDetector` and `RobustnessChecker` to identify common backtesting pitfalls.
- `scripts/chap12_efficient_backtesting.py`: Showcases efficient backtesting techniques, including vectorized and streaming backtesters.
- `scripts/chap19_microstructural_features.py`: Provides an example of how to process high-frequency data and analyze market microstructure features.
- `scripts/chap16_machine_learning_asset_allocation.py`: Demonstrates portfolio optimization using various techniques and performance analysis.
- `scripts/refactored_library_usage.py`: A comprehensive script that ties together multiple library components.

To run an example, execute the following command:
```bash
python3 scripts/chap2_bars.py
```

## Project Structure

```
AFML_ver6/
├── scripts/
│   ├── chap2_bars.py
│   ├── chap3_labeling.py
│   ├── chap4_sample_weights.py
│   ├── chap5_fractional_differentiation.py
│   ├── chap7_cross_validation.py
│   ├── chap8_feature_importance_analysis.py
│   ├── chap8_feature_importance.py
│   ├── chap10_bet_sizing.py
│   ├── chap11_dangers_of_backtesting.py
│   ├── chap12_efficient_backtesting.py
│   ├── chap14_backtest_statistics.py
│   ├── chap16_machine_learning_asset_allocation.py
│   ├── chap19_microstructural_features.py
│   └── refactored_library_usage.py
├── refactored_library/
│   ├── core/
│   ├── data/
│   ├── features/
│   ├── machine_learning/
│   ├── microstructure/
│   ├── portfolio/
│   ├── utils/
│   └── validation/
```

## Core Modules

The library is organized into several core modules, each providing a specific set of functionalities:

-   **`core`**: Contains the main backtesting engines.
    -   `backtester`: Implements `VectorizedBacktester` and other backtesting strategies.
    -   `streaming`: Provides `StreamingBacktester` for event-driven simulations.

-   **`data`**: Handles financial data generation, simulation, and structuring.
    -   `bars`: Functions for creating various bar types.
    -   `labeling`: Implements the Triple Barrier Method.
    -   `weights`: Computes sample weights for financial data.
    -   `simulation`: Generates synthetic financial data for testing.

-   **`features`**: Contains feature engineering tools.
    -   `fractional_differentiation`: Implements fractional differentiation for time series stationarity.

-   **`machine_learning`**: Implements ML algorithms for finance.
    -   `feature_importance`: Provides MDI, MDA, and other feature importance methods.
    -   `ensembling`: Tools for creating ensemble models.

-   **`portfolio`**: Includes tools for portfolio construction, optimization, and analysis.
    -   `optimizer`: Implements various portfolio optimization techniques.
    -   `bet_sizing`: Contains functions for determining position sizes.
    -   `analysis`: Provides in-depth performance and risk analytics.

-   **`validation`**: Provides tools to test the robustness and reliability of models.
    -   `cross_validation`: Implements Purged K-Fold CV.
    -   `statistics`: Computes various backtest performance metrics.
    -   `dangers`: Includes functions to detect common pitfalls in backtesting.
    -   `efficiency`: Helps in analyzing the efficiency of the backtesting process.

-   **`microstructure`**: Includes tools for analyzing market microstructure features.
    -   `processor`: Processes raw tick data into various bar formats.
    -   `analysis`: Provides tools to analyze features from high-frequency data.

-   **`utils`**: Contains various helper functions and utility classes.
    -   `performance`: Monitors the performance of trading strategies.
    -   `visualization`: A set of plotting functions to visualize results.
