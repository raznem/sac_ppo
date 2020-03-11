from collections import defaultdict
from typing import Optional

import numpy as np
import torch.nn as nn

from rltoolkit.logger import get_logger
from rltoolkit.tensorboard_logger import TensorboardWriter


logger = get_logger()


class EvalsWrapper:
    def __init__(
        self,
        Algo: type,
        evals: int,
        tensorboard_dir: str,
        log_all: bool = False,
        *args,
        **kwargs,
    ):
        self.Algo = Algo
        self.evals = evals
        self.args = args
        if log_all:
            kwargs.update({"tensorboard_dir": tensorboard_dir + "/tb_runs"})
        self.kwargs = kwargs
        self.tensorboard_dir = tensorboard_dir + "/tb_hparams"
        self.hparams = {}
        self.metrics = defaultdict(list)
        self.filename = None

    def perform_evaluations(self):
        for i in range(self.evals):
            algo = self.Algo(**self.kwargs)
            if i == 0:
                logger.info("Started %s", algo.filename)
            algo.train()
            self.metrics["frames"].append(algo.stats_logger.frames)
            self.metrics["returns"].append(algo.stats_logger.running_return)
            self.metrics["iterations"].append(algo.iteration)
        self.filename = algo.filename
        self.hparams = algo.hparams
        logger.info("Ended %s", algo.filename)

    def update_tensorboard(self):
        metrics = {}
        for key, val in self.metrics.items():
            key = f"metrics/{key}"
            arr = np.array(val)
            mean = arr.mean()
            std = arr.std()
            metrics[key + "_mean"] = mean
            metrics[key + "_std"] = std

        writer = TensorboardWriter(
            env_name=None,
            log_dir=self.tensorboard_dir,
            filename=self.filename,
            render=False,
        )
        writer.log_hyperparameters(self.hparams, metrics)
