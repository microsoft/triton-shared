import time
import numpy as np
from functools import wraps
import triton
from triton.backends.triton_shared.driver import CPUDriver


def select_cpu_backend():
    triton.runtime.driver.set_active(CPUDriver())


# Unfortunately, we can't use triton.testing.perf_report and triton.testing.do_bench for CPU backend because
# they are very specific to cuda

def measure(repeats=20, percentiles=(20, 50, 90)):
    """
    Decorator to benchmark a function.
    
    Parameters:
    - repeats: int, the number of times the function should be executed for each set of parameters.
    - percentiles: tuple, the percentiles to compute on the execution times.
    
    Returns:
    - A decorated function that prints the average execution time and the requested percentiles.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{func.__name__}{args} {kwargs}, {repeats} times, all results in seconds")
            times = []
            for _ in range(repeats):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)
            
            average_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            computed_percentiles = np.percentile(times, percentiles)
            
            print(f"Avg={average_time:.6f}, min={min_time:.6f},", end=" ")
            for p, value in zip(percentiles, computed_percentiles):
                print(f"{p}pp={value:.6f},", end=" ")
            print(f"max={max_time:.6f}")
            
            return result
        return wrapper
    return decorator