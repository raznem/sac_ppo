from itertools import product

from rltoolkit import A2C_AcM, EvalsWrapper
import multiprocessing as mp

ALGO = A2C_AcM
EVALS = 3
ITERATIONS = 200
RETURN_DONE = 190
GAMMA = [0.99]
A_LR = [3e-3, 1e-2, 3e-2]
C_LR = [1e-3, 3e-3, 1e-2]
BATCH_SIZE = [200]
TENSORBOARD_DIR = "tboard_hyps_test_cartpole_acm"
LOG_ALL = False
LOG_DIR = "tboard_hyps_test_cartpole_acm/basic_logs"
N_CORES = 16
ACM_UPDATE_FREQ = [1, 10, 25]
ACM_LR = [3e-4, 1e-3, 3e-3, 1e-2]


def run_combination(*args, **kwargs):
    evals = EvalsWrapper(*args, **kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()


def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    combinations = product(A_LR, C_LR, BATCH_SIZE, GAMMA, ACM_UPDATE_FREQ, ACM_LR)
    kwargs_list = []
    for a_lr, c_lr, batch_size, gamma, acm_update_freq, acm_lr in combinations:
        kwargs = {
            "Algo": ALGO,
            "evals": EVALS,
            "iterations": ITERATIONS,
            "gamma": gamma,
            "actor_lr": a_lr,
            "critic_lr": c_lr,
            "batch_size": batch_size,
            "tensorboard_dir": TENSORBOARD_DIR,
            "log_all": LOG_ALL,
            "return_done": RETURN_DONE,
            "log_dir": LOG_DIR,
            "verbose": 0,
            "render": False,
            "acm_update_freq": acm_update_freq,
            "acm_lr": acm_lr,
        }
        kwargs_list.append((run_combination, kwargs))

    with mp.Pool(N_CORES) as p:
        p.starmap(apply_kwargs, kwargs_list)
