import numpy as np
import pandas as pd
import os
import gc
import tempfile
import time
from typing import Generator, Callable, Any, List, Optional
from contextlib import contextmanager

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EfficientDataHandler:
    """
    Manages data efficiently for large datasets, using techniques like chunking,
    temporary file storage, and memory-mapping.
    (Derived from chap12.EfficientDataHandler)
    """

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the data handler.

        Args:
            temp_dir: Directory for temporary files. If None, the system's default is used.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_files = []
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        logger.info(f"Initialized data handler with temp directory: {self.temp_dir}")

    def __del__(self):
        """Ensures cleanup of temporary files when the object is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Removes all temporary files created by this handler."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {file_path}: {e}")
        self.temp_files = []

    def chunked_generator(self, df: pd.DataFrame, chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """
        Creates a generator that yields chunks of a DataFrame.

        Args:
            df: The input DataFrame.
            chunk_size: The size of each chunk.

        Yields:
            A chunk of the DataFrame.
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]

    def to_temp_hdf(self, df: pd.DataFrame, key: str = 'data') -> str:
        """
        Saves a DataFrame to a temporary HDF5 file on disk.

        Args:
            df: The DataFrame to save.
            key: The key to use within the HDF5 file.

        Returns:
            The path to the temporary file.
        """
        temp_path = os.path.join(self.temp_dir, f"temp_data_{time.time_ns()}.h5")
        df.to_hdf(temp_path, key=key, mode='w', format='table')
        self.temp_files.append(temp_path)
        logger.debug(f"Saved DataFrame (shape: {df.shape}) to temporary HDF5 file: {temp_path}")
        return temp_path

    def from_temp_hdf(self, file_path: str, key: str = 'data') -> pd.DataFrame:
        """Loads a DataFrame from a temporary HDF5 file."""
        return pd.read_hdf(file_path, key=key)

    def to_memmap(self, array: np.ndarray) -> np.memmap:
        """
        Creates a memory-mapped version of a NumPy array on disk. This is useful
        for sharing large arrays between processes without copying into RAM.

        Args:
            array: The NumPy array to memory-map.

        Returns:
            A memory-mapped NumPy array.
        """
        temp_path = os.path.join(self.temp_dir, f"temp_array_{time.time_ns()}.mmap")
        memmapped_array = np.memmap(temp_path, dtype=array.dtype, mode='w+', shape=array.shape)
        memmapped_array[:] = array[:]
        memmapped_array.flush()
        self.temp_files.append(temp_path)
        logger.debug(f"Created memory-mapped array (shape: {array.shape}) at: {temp_path}")
        # Return a read-only view
        return np.memmap(temp_path, dtype=array.dtype, mode='r', shape=array.shape)

    def process_in_chunks(self, df: pd.DataFrame,
                          process_func: Callable[[pd.DataFrame], Any],
                          chunk_size: int = 50000,
                          combine_func: Optional[Callable[[List], Any]] = pd.concat) -> Any:
        """
        Processes a large DataFrame in smaller chunks to conserve memory.

        Args:
            df: The input DataFrame.
            process_func: A function to apply to each chunk.
            chunk_size: The size of each chunk.
            combine_func: A function to combine the results from each chunk.
                          Defaults to `pd.concat`.

        Returns:
            The combined result from processing all chunks.
        """
        results = [process_func(chunk) for chunk in self.chunked_generator(df, chunk_size)]
        if combine_func:
            return combine_func(results)
        return results 