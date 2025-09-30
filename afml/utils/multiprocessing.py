import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from typing import Union, Callable, List, Dict, Any, Generator


def _process_chunk(func: Callable, chunk: Union[pd.DataFrame, pd.Series], **kwargs) -> Any:
    """Helper function to apply a function to a chunk of a pandas object."""
    return chunk.apply(func, **kwargs)

def mp_pandas_obj(
    func: Callable,
    pd_obj: Union[pd.DataFrame, pd.Series],
    n_jobs: int = -1,
    axis: int = 0,
    **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """
    Applies a function to a pandas DataFrame or Series in parallel using joblib.
    This is a general-purpose implementation of the concept described in Chapter 20.

    Args:
        func: The function to apply. It will be applied to each row (or column).
        pd_obj: The pandas DataFrame or Series to process.
        n_jobs: The number of jobs to run in parallel. -1 means using all available CPUs.
        axis: The axis to apply the function on (0 for columns, 1 for rows).
        **kwargs: Additional keyword arguments to pass to the function `func`.

    Returns:
        A pandas DataFrame or Series with the combined results.
    """
    if axis == 1:
        # Split DataFrame by rows
        chunks = np.array_split(pd_obj, n_jobs if n_jobs > 0 else os.cpu_count())
    else:
        # Split DataFrame by columns
        chunks = [pd_obj.iloc[:, i::n_jobs] for i in range(n_jobs if n_jobs > 0 else os.cpu_count())]

    # Use joblib to process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(chunk.apply)(func, axis=axis, **kwargs) for chunk in chunks
    )

    # Combine the results
    if isinstance(pd_obj, pd.DataFrame):
        if axis == 1:
            return pd.concat(results, axis=0)
        else:
            return pd.concat(results, axis=1)
    elif isinstance(pd_obj, pd.Series):
        return pd.concat(results)

    return pd.concat(results)
