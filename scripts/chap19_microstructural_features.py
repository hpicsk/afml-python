import matplotlib.pyplot as plt

# Import from the refactored library
from afml.data.hf_simulation import HighFrequencyDataSimulator
from afml.microstructure.processor import TickDataProcessor
from afml.microstructure.analysis import MarketMicrostructureAnalyzer

def main():
    """
    Example script to demonstrate market microstructure analysis using the refactored library.
    """
    print("--- Market Microstructure Analysis Demonstration ---")
    
    # 1. Simulate Data
    print("\n1. Generating synthetic trade data...")
    simulator = HighFrequencyDataSimulator(seed=42)
    trades_df = simulator.generate_trades(n_trades=10000, initial_price=150.0)
    print("Sample of generated trades:")
    print(trades_df.head())
    
    # 2. Process Data
    print("\n2. Processing tick data...")
    processor = TickDataProcessor()
    
    # Clean trades
    cleaned_trades = processor.clean_trades(trades_df)
    
    # Create tick bars
    tick_bars = processor.create_tick_bars(cleaned_trades, ticks_per_bar=500)
    print("\nSample of 500-tick bars:")
    print(tick_bars.head())

    # 3. Analyze Microstructure
    print("\n3. Analyzing microstructure features...")
    analyzer = MarketMicrostructureAnalyzer()
    
    # Calculate Kyle's Lambda (price impact)
    print("\nCalculating Kyle's Lambda...")
    lambda_val, model_summary = analyzer.calculate_kyle_lambda(cleaned_trades, window='1T')
    print(f"\nEstimated Kyle's Lambda: {lambda_val}")
    print("Regression Summary for Lambda Calculation:")
    print(model_summary)
    
    # 4. Visualization
    print("\n4. Visualizing results...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot price and volume
    axes[0].plot(cleaned_trades.index, cleaned_trades['price'], label='Trade Price', alpha=0.7)
    ax0_twin = axes[0].twinx()
    ax0_twin.bar(cleaned_trades.index, cleaned_trades['volume'], width=0.001, alpha=0.2, color='gray', label='Trade Volume')
    axes[0].set_title('Simulated Trade Data')
    axes[0].set_ylabel('Price')
    ax0_twin.set_ylabel('Volume')
    axes[0].legend(loc='upper left')
    ax0_twin.legend(loc='upper right')
    
    # Plot tick bars
    axes[1].plot(tick_bars.index, tick_bars['close'], marker='o', linestyle='-', label='Tick Bar Close Price')
    axes[1].set_title('500-Tick Bars')
    axes[1].set_ylabel('Price')
    axes[1].set_xlabel('Time')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()