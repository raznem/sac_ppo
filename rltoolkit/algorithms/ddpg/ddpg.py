import torch
from torch.nn import functional as F
import copy
from itertools import chain
import numpy as np

from rltoolkit import config
from rltoolkit.algorithms.ddpg.models import Actor, Critic
from rltoolkit.buffer import ReplayBuffer
from rltoolkit.rl import RL
from rltoolkit.utils import measure_time
from rltoolkit.logger import get_logger

logger = get_logger()


class DDPG(RL):
    def __init__(
        self,
        actor_lr: float = config.DDPG_LR,
        critic_lr: float = config.DDPG_LR,
        tau: float = config.TAU,
        update_batch_size: int = config.UPDATE_BATCH_SIZE,
        buffer_size: int = config.BUFFER_SIZE,
        random_frames: int = config.RANDOM_FRAMES,
        update_freq: int = config.UPDATE_FREQ,
        grad_steps: int = config.GRAD_STEPS,
        act_noise: float = config.ACT_NOISE,
        *args,
        **kwargs,
    ):
        f"""Deep Deterministic Policy Gradient implementation

        Args:
            actor_lr (float, optional): Learning rate of the actor.
                Defaults to { config.DDPG_LR }.
            critic_lr (float, optional): Learning rate of the critic.
                Defaults to { config.DDPG_LR }.
            tau (float, optional): Tau coefficient for polyak averaging.
                Defaults to { config.TAU }.
            update_batch_size (int, optional): Batch size for gradient step.
                Defaults to { config.UPDATE_BATCH_SIZE }.
            buffer_size (int, optional): Size of replay buffer.
                Defaults to { config.BUFFER_SIZE }.
            random_frames (int, optional): Number of frames with random actions at
                the beggining. Defaults to { config.RANDOM_FRAMES }.
            update_freq (int, optional): Freqency of SAC updates (in frames).
                Defaults to { config.UPDATE_FREQ }.
            grad_steps (int, optional): Number of SAC updates for one step.
                Defaults to { config.GRAD_STEPS }.
            act_noise (float, optional): Actions noise multiplier.
                Defaults to { config.ACT_NOISE }.
            env_name (str, optional): Name of the gym environment.
                Defaults to { config.ENV_NAME }.
            gamma (float, optional): Discount factor. Defaults to { config.GAMMA }.
            stats_freq (int, optional): Frequency of logging the progress.
                Defaults to { config.STATS_FREQ }.
            batch_size (int, optional): Number of frames used for one algorithm step
                (could be higher because batch collection stops when rollout ends).
                Defaults to { config.BATCH_SIZE }.
            iterations (int, optional): Number of algorithms iterations.
                Defaults to { config.ITERATIONS }.
            max_frames (int, optional): Limit of frames for training.
                Defaults to { None }.
            return_done (Union[int, None], optional): target return, which will stop
                training if reached. Defaults to { config.RETURN_DONE }.
            log_dir (str, optional): Path for basic logs which includes final model.
                Defaults to { config.LOG_DIR }.
            use_gpu (bool, optional): Use CUDA. Defaults to { config.USE_GPU }.
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
            verbose (int, optional): Verbose level. Defaults to { config.VERBOSE }.
            render (bool, optional): Render rollouts to tensorboard.
                Defaults to { config.RENDER }.

        """
        super().__init__(*args, **kwargs)
        self._actor = None
        self.actor_optimizer = None
        self._actor_targ = None
        self._critic = None
        self.critic_optimizer = None
        self.critic_targ = None

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.update_batch_size = update_batch_size
        self.buffer_size = buffer_size
        self.random_frames = random_frames
        self.update_freq = update_freq
        self.grad_steps = grad_steps
        self.act_noise = act_noise

        self.opt = torch.optim.Adam
        self.actor = Actor(self.ob_dim, self.ac_lim, self.ac_dim)
        self.critic = Critic(self.ob_dim, self.ac_dim)

        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.ob_dim,
            self.ac_dim,
            dtype=torch.float32,
            discrete=self.discrete,
        )

        self.loss = {"actor": 0.0, "critic": 0.0}
        new_hparams = {
            "hparams/actor_lr": self.actor_lr,
            "hparams/critic_lr": self.critic_lr
            "hparams/tau": self.tau,
            "hparams/update_batch_size": self.update_batch_size,
            "hparams/buffer_size": self.buffer_size,
            "hparams/random_frames": self.random_frames,
            "hparams/update_freq": self.update_freq,
            "hparams/grad_steps": self.grad_steps,
            "hparams/act_noise": self.act_noise,
        }
        self.hparams.update(new_hparams)

    def set_model(self, model, lr):
        model.to(device=self.device)
        optimizer = self.opt(model.parameters(), lr=lr)
        return model, optimizer

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, model: torch.nn.Module):
        self._actor, self.actor_optimizer = self.set_model(model, self.actor_lr)
        self.actor_targ = copy.deepcopy(self._actor)
        for p in self.actor_targ.parameters():
            p.requires_grad = False

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, model: torch.nn.Module):
        self._critic, self.critic_optimizer = self.set_model(model, self.critic_lr)
        self.critic_targ = copy.deepcopy(self._critic)
        for p in self.critic_targ.parameters():
            p.requires_grad = False

    @measure_time
    def perform_iteration(self):
        f"""Single train step of algorithm

        Returns:
            Memory: Buffer filled with one batch
            float: Time taken for evaluation
        """
        self.collect_batch_and_train()
        return self.replay_buffer.last_rollout()

    def noise_action(self, obs, act_noise):
        action, _ = self._actor.act(obs)
        action += act_noise * torch.randn(self.ac_dim)
        return np.clip(action, -self.ac_lim, self.ac_lim)

    def collect_batch_and_train(self):
        f"""Perform full rollouts and collect samples till batch_size number of steps
            will be added to the replay buffer

        """
        collected = 0
        while collected < self.batch_size:
            self.stats_logger.rollouts += 1

            obs = self.env.reset()
            done = False
            obs = self.process_obs(obs)
            prev_idx = self.replay_buffer.add_obs(obs)

            while not done:
                if self.stats_logger.frames < self.random_frames:
                    action = torch.tensor(self.env.action_space.sample()).unsqueeze(0)
                else:
                    action = self.noise_action(obs, self.act_noise)
                action_proc = self.process_action(action, obs)
                obs, rew, done, _ = self.env.step(action_proc)
                obs = self.process_obs(obs)
                next_idx = self.replay_buffer.add_obs(obs)
                self.replay_buffer.add_timestep(prev_idx, next_idx, action, rew, done)
                prev_idx = next_idx
                self.stats_logger.frames += 1
                collected += 1

                if (
                    len(self.replay_buffer) > self.update_batch_size
                    and self.stats_logger.frames % self.update_freq == 0
                ):
                    for _ in range(self.grad_steps):
                        self.update()

    def compute_qfunc_targ(
        self, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor
    ):
        """Compute targets for Q-functions

        Args:
            reward (torch.Tensor): batch of rewards
            next_obs (torch.Tensor): batch of next observations
            done (torch.Tensor): batch of done

        Returns:
            torch.Tensor: Q-function targets for the batch
        """
        with torch.no_grad():
            next_action, _ = self.actor_targ(next_obs)
            q_target = self.critic_targ(next_obs, next_action)

            qfunc_target = reward + self.gamma * (1 - done) * q_target

        return qfunc_target

    def compute_pi_loss(self, obs):
        """Loss for the policy

        Args:
            obs (torch.Tensor): batch of observations

        Returns:
            torch.Tensor: policy loss
        """
        action, _ = self._actor(obs)
        loss = -self._critic(obs, action).mean()
        return loss

    def update_target_nets(self):
        """Update target networks with Polyak averaging
        """
        with torch.no_grad():
            # Polyak averaging:
            learned_params = chain(self._critic.parameters(), self._actor.parameters())
            targets_params = chain(
                self.critic_targ.parameters(), self.actor_targ.parameters()
            )
            for params, targ_params in zip(learned_params, targets_params):
                targ_params.data.mul_(1 - self.tau)
                targ_params.data.add_((self.tau) * params.data)

    def update(self):
        """DDPG update:
        """
        obs, next_obs, action, reward, done = self.replay_buffer.sample_batch(
            self.update_batch_size
        )

        y = self.compute_qfunc_targ(reward, next_obs, done)

        # Update Q-function by one step
        y_q = self._critic(obs, action)
        loss_q = F.mse_loss(y_q, y)

        self.loss["critic"] = loss_q.item()

        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        # Update policy by one step
        self._critic.eval()

        loss = self.compute_pi_loss(obs)
        self.loss["actor"] = loss.item()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Update target networks

        self.update_target_nets()

        self._critic.train()

    def save_model(self, save_path=None):
        if self.filename is None and save_path is None:
            raise AttributeError
        elif save_path is None:
            save_path = str(self.log_path)

        torch.save(self._actor.state_dict(), save_path + "_actor_model.pt")
        torch.save(self._critic.state_dict(), save_path + "_critic_model.pt")

    def process_obs(self, obs):
        f"""Pre-processing of observation before it will go to the policy

        Args:
            obs (iter): original observation from env

        Returns:
            torch.Tensor: processed observation
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = torch.unsqueeze(obs, dim=0)
        return obs

    def process_action(self, action: torch.Tensor, obs: torch.tensor, *args, **kwargs):
        f"""Pre-processing of action before it will go the env.
        It will not be saved to the buffer.

        Args:
            action (torch.Tensor): action from the policy
            obs (torch.tensor): observations for this actions

        Returns:
            np.array: processed action
        """
        action = action.cpu().numpy()[0]
        return action

    def test(self, episodes=None):
        f"""Run deterministic policy and log average return

        Args:
            episodes (int, optional): Number of episodes for test. Defaults to { 10 }.

        Returns:
            float: mean episode reward
        """
        if episodes is None:
            episodes = self.test_episodes
        returns = []
        for j in range(episodes):
            obs = self.env.reset()
            done = False
            ep_ret = 0
            while not done:
                obs = self.process_obs(obs)
                action, _ = self._actor.act(obs, deterministic=True)
                action_proc = self.process_action(action, obs)
                obs, r, done, _ = self.env.step(action_proc)
                ep_ret += r
            returns.append(ep_ret)

        return np.mean(returns)


if __name__ == "__main__":
    model = DDPG(
        env_name="HalfCheetah-v2",
        actor_lr=1e-3,
        critic_lr=1e-3,
        iterations=1000,
        gamma=0.95,
        batch_size=1000,
        buffer_size=int(1e6),
        update_batch_size=256,
        random_frames=10000,
        stats_freq=5,
        update_freq=50,
        grad_steps=50,
        act_noise=0.1,
        tensorboard_dir="logs",
        # tensorboard_comment="pi_upd_each",
        test_episodes=1,
    )
    model.train()
