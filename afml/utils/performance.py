import time
import psutil
import os
import gc
from contextlib import contextmanager
from typing import Dict, Any

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    A utility class to monitor and report on the performance (time and memory)
    of backtesting operations.
    (Derived from chap12.BudgetBacktesterBase)
    """
    def __init__(self, memory_limit_gb: float = None):
        """
        Initialize the monitor.

        Args:
            memory_limit_gb: Optional memory limit in GB. If None, it defaults to 80% of available RAM.
        """
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.memory_limit_gb = memory_limit_gb or (available_memory * 0.80)
        self.metrics = {'time': {}, 'memory': {}}
        logger.info(f"Performance monitor initialized. Memory limit set to {self.memory_limit_gb:.2f} GB.")

    @staticmethod
    def get_memory_usage() -> float:
        """Returns current process memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    @contextmanager
    def timer(self, operation_name: str):
        """
        A context manager to time an operation and log its memory change.

        Args:
            operation_name: A descriptive name for the operation being timed.
        """
        start_time = time.time()
        start_mem = self.get_memory_usage()
        
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            end_mem = self.get_memory_usage()
            mem_change = end_mem - start_mem
            
            self.metrics['time'].setdefault(operation_name, []).append(elapsed_time)
            self.metrics['memory'].setdefault(operation_name, []).append(mem_change)
            
            logger.debug(f"'{operation_name}' took {elapsed_time:.4f}s. Memory change: {mem_change:+.4f} GB.")

    def check_memory_threshold(self, threshold_ratio: float = 0.9) -> bool:
        """
        Checks if the current memory usage is approaching the defined limit.

        Args:
            threshold_ratio: The fraction of the memory limit to check against.

        Returns:
            True if memory usage is below the threshold, False otherwise.
        """
        current_usage = self.get_memory_usage()
        if current_usage > self.memory_limit_gb * threshold_ratio:
            logger.warning(f"Memory alert! Usage: {current_usage:.2f} GB, Limit: {self.memory_limit_gb:.2f} GB.")
            self.force_garbage_collection()
            return False
        return True

    def force_garbage_collection(self):
        """Forces a garbage collection cycle."""
        with self.timer("garbage_collection"):
            gc.collect()

    def report_performance(self) -> Dict[str, Any]:
        """Generates a summary report of all timed operations."""
        report = {}
        for op, times in self.metrics['time'].items():
            report[op] = {
                'total_time_s': sum(times),
                'avg_time_s': sum(times) / len(times),
                'calls': len(times),
                'avg_mem_change_gb': sum(self.metrics['memory'].get(op, [0])) / len(times)
            }
        logger.info("--- Performance Report ---")
        for op, data in report.items():
            logger.info(f"{op}: {data['calls']} calls, {data['total_time_s']:.2f}s total, {data['avg_mem_change_gb']:+.3f} GB avg mem change.")
        return report 