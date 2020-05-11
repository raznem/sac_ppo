from rltoolkit import (
    DDPG,
    PPO,
    SAC,
    DDPG_AcM,
    EvalsWrapper,
    EvalsWrapperACM,
    PPO_AcM,
    SAC_AcM,
)

ITERATIONS = 3
MAX_FRAMES = 2e2
STATS_FREQ = 1
EVALS = 2
ENV = "Pendulum-v0"
BATCH_SIZE = 100
ACM_PRETRAIN_SAMPLES = 10
ACM_PRETRAIN_EPOCHS = 2
OFF_POLICY_BUFFER_SIZE = 100


def test_ppo_evals():
    evals_wrapper = EvalsWrapper(
        Algo=PPO,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        ppo_batch_size=32,
        max_ppo_epochs=2,
        env_name=ENV,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_ppo_acm_evals():
    evals_wrapper = EvalsWrapperACM(
        Algo=PPO_AcM,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        ppo_batch_size=32,
        max_ppo_epochs=2,
        acm_pre_train_samples=ACM_PRETRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRETRAIN_EPOCHS,
        env_name=ENV,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_ddpg_evals():
    evals_wrapper = EvalsWrapper(
        Algo=DDPG,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_ddpg_acm_evals():
    evals_wrapper = EvalsWrapperACM(
        Algo=DDPG_AcM,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        acm_pre_train_samples=ACM_PRETRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRETRAIN_EPOCHS,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_sac_evals():
    evals_wrapper = EvalsWrapper(
        Algo=SAC,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()


def test_sac_acm_evals():
    evals_wrapper = EvalsWrapperACM(
        Algo=SAC_AcM,
        evals=EVALS,
        tensorboard_dir="test_logs",
        max_frames=MAX_FRAMES,
        iterations=ITERATIONS,
        stats_freq=STATS_FREQ,
        batch_size=BATCH_SIZE,
        acm_pre_train_samples=ACM_PRETRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRETRAIN_EPOCHS,
        env_name=ENV,
        buffer_size=OFF_POLICY_BUFFER_SIZE,
    )
    evals_wrapper.perform_evaluations()
    evals_wrapper.update_tensorboard()
