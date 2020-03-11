import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from scipy.stats import entropy


def measure_time(func: Callable) -> Callable:
    def decorated(*args, **kwargs):
        time = datetime.datetime.now()
        result = func(*args, **kwargs)
        time_after = datetime.datetime.now()
        time_diff = time_after - time
        return result, time_diff.total_seconds()

    return decorated


def get_pretty_type_name(item: Any) -> str:
    t = str(type(item)).split("'")[1].split(".")[-1]
    return t


def get_log_dir(log_dir: str) -> Path:
    """
    Get directory name for new RL experiment.

    Arguments:
        log_dir {str} -- absolute or relative path to parent directory.

    Returns:
        Path -- new experiment tensorboard event path in the following form:
            "/path/to/log/dir/ + time"
    """
    log_dir = Path(log_dir)
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = log_dir / current_time
    return log_dir


def get_time() -> str:
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    return current_time


def kl_divergence(log_p: torch.tensor, log_q: torch.tensor) -> torch.tensor:
    """
    Calculate KL divergence of two distributions p and q.

    Args:
        p (torch.tensor): log probabilites from distribution 1
        q (torch.tensor): log probabilites from distribution 2

    Returns:
        torch.tensor: KL divergence >= 0
    """
    p_new = torch.exp(log_p.detach()) + 1.2e-7
    q_new = torch.exp(log_q.detach()) + 1.2e-7

    return entropy(p_new, q_new)
