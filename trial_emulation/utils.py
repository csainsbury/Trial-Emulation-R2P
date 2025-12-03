"""
Utility functions for conditional printing, checks, and helpers.
"""

from typing import Any, Optional
import sys


def quiet_print(quiet: bool, x: Any, **kwargs) -> Optional[Any]:
    """
    Conditional printing.
    
    Args:
        quiet: If False, print x
        x: Object to print
        **kwargs: Passed to print function
        
    Returns:
        Value of print(...) or None
    """
    if not quiet:
        return print(x, **kwargs)
    return None


def quiet_msg(quiet: bool, x: str, **kwargs) -> None:
    """
    Conditional messages.
    
    Args:
        quiet: If False, print message
        x: Message to print
        **kwargs: Passed to message function
    """
    if not quiet:
        print(x, file=sys.stderr, **kwargs)


def quiet_line(quiet: bool) -> None:
    """
    Print a line separator.
    
    Args:
        quiet: If False, print line
    """
    if not quiet:
        width = 80  # Default console width
        try:
            import shutil
            width = shutil.get_terminal_size().columns
        except:
            pass
        quiet_msg(quiet, "-" * int(0.75 * width))


def console_line(prop: float = 0.75) -> str:
    """
    Generate a console line separator.
    
    Args:
        prop: Proportion of terminal width to use
        
    Returns:
        String of dashes
    """
    width = 80
    try:
        import shutil
        width = shutil.get_terminal_size().columns
    except:
        pass
    return "-" * int(prop * width)


def assert_monotonic(x: list, increasing: bool = True) -> None:
    """
    Assert monotonicity of a numeric vector.
    
    Args:
        x: Numeric vector
        increasing: Test for increasing (True) or decreasing (False)
        
    Raises:
        ValueError: If not monotonic
    """
    if increasing:
        if not all(x[i] <= x[i+1] for i in range(len(x)-1)):
            raise ValueError("Not monotonically increasing")
    else:
        if not all(x[i] >= x[i+1] for i in range(len(x)-1)):
            raise ValueError("Not monotonically decreasing")


def p99_weight(weight: list) -> list:
    """
    Truncate weights at the 1st and 99th percentile.
    
    Args:
        weight: List of weights
        
    Returns:
        Truncated weights
    """
    import numpy as np
    p99 = np.percentile(weight, [1, 99])
    return limit_weight(weight, p99[0], p99[1])


def limit_weight(weight: list, lower_limit: float, upper_limit: float) -> list:
    """
    Truncate weights using user-defined limits.
    
    Args:
        weight: List of weights
        lower_limit: Minimum possible weight
        upper_limit: Maximum possible weight
        
    Returns:
        Truncated weights
    """
    import numpy as np
    weight = np.array(weight)
    weight[weight > upper_limit] = upper_limit
    weight[weight < lower_limit] = lower_limit
    return weight.tolist()

