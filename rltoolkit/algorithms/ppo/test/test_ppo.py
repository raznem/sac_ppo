import pytest
import torch

from rltoolkit.algorithms.ppo.ppo import PPO


def test_cartpole():
    env_name = "CartPole-v0"
    iterations = 100
    stats_freq = 10
    return_done = 50
    model = PPO(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
        return_done=return_done,
    )
    model.train()


CLIP_TEST_CASES = [
    (
        torch.tensor([-2.3, -5, -1.4, -1.5], requires_grad=True),
        torch.tensor([-2.3, -5, -1.4, -1.5], requires_grad=True),
        torch.tensor([1, 2.0, 3.0, 4.0]),
        -2.5,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-1]),
        1,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([-1.0]),
        0.8,
    ),
    (
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([1.0]),
        -1.2,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([1.0]),
        -0.3679,
    ),
]


@pytest.mark.parametrize(
    "action_logprobs, new_logprobs, advantages, expected_result", CLIP_TEST_CASES
)
def test_clip_loss(action_logprobs, new_logprobs, advantages, expected_result):
    ppo = PPO(epsilon=0.2)
    result = ppo._clip_loss(action_logprobs, new_logprobs, advantages)

    assert expected_result == pytest.approx(result.item(), 0.0001)
    assert result.requires_grad
