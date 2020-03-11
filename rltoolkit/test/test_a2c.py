from rltoolkit import A2C, SAC
import tempfile


def test_cartpole():
    env_name = "CartPole-v0"
    iterations = 100
    stats_freq = 10
    return_done = 50
    model = A2C(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
        return_done=return_done,
    )
    model.train()


def test_pendulum():
    env_name = "Pendulum-v0"
    iterations = 4
    stats_freq = 1
    model = A2C(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
    )
    model.train()


def test_tensorboard():
    iterations = 4
    stats_freq = 1
    tmpdir = tempfile.TemporaryDirectory()
    tensorboard_dir = tmpdir.name
    model = A2C(
        iterations=iterations, stats_freq=stats_freq, tensorboard_dir=tensorboard_dir
    )
    model.train()
    tmpdir.cleanup()


def test_mujoco_reacher():
    env_name = "Reacher-v2"
    iterations = 100
    stats_freq = 20
    return_done = -20
    model = A2C(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
        return_done=return_done,
    )
    model.train()


def test_sac():
    env_name = "Pendulum-v0"
    iterations = 10
    stats_freq = 2
    return_done = -1000
    model = SAC(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
        return_done=return_done,
    )
    model.train()
