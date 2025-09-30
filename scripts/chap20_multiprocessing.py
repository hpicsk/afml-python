import numpy as np
import pandas as pd
import time

from afml.utils.multiprocessing import mp_pandas_obj

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_heavy_calculation(row: pd.Series) -> float:
    """
    A sample function that simulates a heavy calculation on a row.
    For example, fitting a model or running a complex simulation.
    """
    # Simulate work by performing some math operations
    val = 0
    for i in range(len(row)):
        val += np.sqrt(abs(row.iloc[i] * 100)) / (i + 1)
    time.sleep(0.001) # Simulate I/O or other blocking calls
    return val

def main():
    """
    Example script demonstrating the use of the multiprocessing utility
    from Chapter 20.
    """
    logger.info("--- Multiprocessing Demonstration (Chapter 20) ---")

    # 1. Create a large synthetic dataset
    n_rows = 5000
    n_cols = 10
    logger.info(f"\n[1] Generating a synthetic DataFrame with {n_rows} rows and {n_cols} columns...")
    df = pd.DataFrame(np.random.randn(n_rows, n_cols),
                      columns=[f'feature_{i}' for i in range(n_cols)])

    # 2. Single-threaded execution using standard pandas apply
    logger.info("\n[2] Running calculation using a standard single-threaded apply...")
    start_time_single = time.time()
    # We only apply to the first 100 rows for a quick comparison
    single_thread_result = df.head(100).apply(simulate_heavy_calculation, axis=1)
    end_time_single = time.time()
    duration_single = end_time_single - start_time_single
    logger.info(f"Single-threaded execution on 100 rows took: {duration_single:.4f} seconds.")

    # 3. Multi-threaded execution using mp_pandas_obj
    logger.info("\n[3] Running the same calculation using mp_pandas_obj in parallel...")
    start_time_multi = time.time()
    multi_thread_result = mp_pandas_obj(
        func=simulate_heavy_calculation,
        pd_obj=df,
        axis=1,
        n_jobs=-1 # Use all available cores
    )
    end_time_multi = time.time()
    duration_multi = end_time_multi - start_time_multi
    logger.info(f"Multi-threaded execution on {n_rows} rows took: {duration_multi:.4f} seconds.")

    # 4. Verification
    logger.info("\n[4] Verifying results...")
    # Re-run single-threaded on the same data as multi-threaded for a fair comparison
    logger.info("Running single-threaded on the full dataset for verification (this may take a while)...")
    start_time_single_full = time.time()
    single_thread_result_full = df.apply(simulate_heavy_calculation, axis=1)
    end_time_single_full = time.time()
    logger.info(f"Full single-threaded execution took: {end_time_single_full - start_time_single_full:.4f} seconds.")

    if np.allclose(single_thread_result_full, multi_thread_result):
        logger.info("Results from single-threaded and multi-threaded executions are identical.")
    else:
        logger.error("Mismatch found between single-threaded and multi-threaded results!")

    logger.info(f"\nSpeedup factor: {(end_time_single_full - start_time_single_full) / duration_multi:.2f}x")

if __name__ == "__main__":
    main()
