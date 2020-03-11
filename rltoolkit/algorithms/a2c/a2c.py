import torch
import numpy as np

from rltoolkit import config
from rltoolkit.basic_model import Actor, Critic
from rltoolkit.buffer import Memory
from rltoolkit.rl import RL
from rltoolkit.utils import measure_time
from rltoolkit.logger import get_logger

logger = get_logger()


class A2C(RL):
    def __init__(
        self,
        actor_lr: float = config.A_LR,
        critic_lr: float = config.C_LR,
        critic_num_target_updates: int = config.NUM_TARGET_UPDATES,
        num_critic_updates_per_target: int = config.NUM_CRITIC_UPDATES,
        normalize_adv: bool = config.NORMALIZE_ADV,
        *args,
        **kwargs,
    ):
        f"""A2C implementation.

        Args:
            actor_lr (float, optional): Learning rate of the actor.
                Defaults to { config.A_LR }.
            critic_lr (float, optional): Learning rate of the critic.
                Defaults to { config.C_LR }.
            critic_num_target_updates (int, optional): Number of target q-value updates
                per iteration. Defaults to { config.NUM_TARGET_UPDATES }.
            num_critic_updates_per_target (int, optional): Number of gradient steps of
                critic for one target. Defaults to { config.NUM_CRITIC_UPDATES }.
            normalize_adv (bool, optional): Normalize advantages for actor.
                Defaults to { config.NORMALIZE_ADV }.
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
            max_frames (int, optional): Limit of frames for training. Defaults to
                { None }.
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
        self._critic = None
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.critic_num_target_updates = critic_num_target_updates
        self.num_critic_updates_per_target = num_critic_updates_per_target
        self.normalize_adv = normalize_adv

        self.opt = torch.optim.Adam
        self.actor = Actor(self.ob_dim, self.ac_lim, self.ac_dim, self.discrete)
        self.critic = Critic(self.ob_dim)

        self.loss = {"actor": 0.0, "critic": 0.0}
        new_hparams = {
            "hparams/actor_lr": self.actor_lr,
            "hparams/critic_lr": self.critic_lr,
            "hparams/critic_num_target_updates": self.critic_num_target_updates,
            "hparams/num_critic_updates_per_target": self.num_critic_updates_per_target,
        }
        self.hparams.update(new_hparams)

    @property
    def actor(self):
        return self._actor

    @actor.setter
    def actor(self, model: torch.nn.Module):
        self._actor = model
        self._actor.to(device=self.device)
        self.actor_optimizer = self.opt(self._actor.parameters(), lr=self.actor_lr)

    @property
    def critic(self):
        return self._critic

    @critic.setter
    def critic(self, model: torch.nn.Module):
        self._critic = model
        self._critic.to(device=self.device)
        self.critic_optimizer = self.opt(self._critic.parameters(), lr=self.critic_lr)

    @measure_time
    def perform_iteration(self):
        f"""Single train step of algorithm

        Returns:
            Memory: Buffer filled with one batch
            float: Time taken for evaluation
        """
        buffer = Memory()
        self.collect_batch(buffer)
        advantages = self.update_critic(buffer)
        self.update_actor(advantages, buffer)
        return buffer

    def collect_batch(self, buffer: Memory):
        f"""Perform full rollouts and collect samples till it will be filled with
            more than batch_size number.

        Args:
            buffer (Memory): Memory to append new samples.

        Returns:
            Memory: Extended buffer
        """
        start_buffer_len = len(buffer)
        while len(buffer) < self.batch_size:
            self.stats_logger.rollouts += 1

            obs = self.env.reset()
            done = False
            obs = self.process_obs(obs)
            prev_idx = buffer.add_obs(obs)

            while not done:
                action, action_logprobs = self.actor.act(obs)
                action_proc = self.process_action(action, obs)
                obs, rew, done, _ = self.env.step(action_proc)
                obs = self.process_obs(obs)
                next_idx = buffer.add_obs(obs)
                buffer.add_timestep(
                    prev_idx, next_idx, action, action_logprobs, rew, done
                )
                prev_idx = next_idx
            buffer.end_rollout()

        num_new_obs = len(buffer) - start_buffer_len
        self.stats_logger.frames += num_new_obs
        return buffer

    def update_critic(self, buffer: Memory):
        f"""One iteration of critic update

        Args:
            buffer (Memory): memory with samples

        Returns:
            torch.Tensor: tensor containing advantages for observations from buffer
        """
        obs = torch.stack(buffer.obs)
        next_obs = torch.stack(buffer.next_obs)
        reward = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        done = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)

        self.loss["critic"] = 0.0

        for _ in range(self.critic_num_target_updates):
            next_state_value = self.critic(next_obs)
            next_state_value = next_state_value.detach()
            q_val = reward + self.gamma * (1 - done) * next_state_value.squeeze()

            for _ in range(self.num_critic_updates_per_target):
                state_value = self.critic(obs)
                advantage = q_val - state_value.squeeze()

                critic_loss = 0.5 * advantage.pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.loss["critic"] += critic_loss.item()

        self.loss["critic"] /= (
            self.critic_num_target_updates * self.num_critic_updates_per_target
        )
        # WARNING: should be changed in case of addition of dropout or other
        # mode-dependent layers

        advantages = self.calculate_advantage(buffer)

        return advantages

    def calculate_advantage(self, buffer: Memory) -> torch.tensor:
        """
        Estimate advantage using critic and rewards.

        Args:
            buffer (Memory): memory with samples

        Returns:
            torch.tensor: tensor containing advantages for observations from buffer
        """
        obs = torch.stack(buffer.obs)

        with torch.no_grad():
            q_val = self.calculate_q_val(buffer)

            state_value = self.critic(obs)
            advantages = q_val - state_value.squeeze()

        return advantages

    def calculate_q_val(self, buffer: Memory) -> torch.tensor:
        """
        Use critic to predict Q value of given states.

        Args:
            buffer (Memory): memory with samples

        Returns:
            torch.tensor: tensor containing Q values for observations from buffer
        """
        next_obs = torch.stack(buffer.next_obs)
        reward = torch.tensor(buffer.rewards, dtype=torch.float32, device=self.device)
        done = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)

        next_state_value = self.critic(next_obs)
        next_state_value = next_state_value.detach()
        q_val = reward + self.gamma * (1 - done) * next_state_value.squeeze()
        return q_val

    def update_actor(self, advantages: torch.Tensor, buffer: Memory):
        f"""One iteration of actor update

        Args:
            advantages (torch.Tensor): advantages for observations from buffer
            buffer (Memory): buffer with samples
        """
        if self.normalize_adv:
            advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-8
            )

        action_logprobs = buffer.action_logprobs
        action_logprobs = torch.cat(action_logprobs)

        actor_loss = (-action_logprobs * advantages).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.loss["actor"] = actor_loss.item()

    def save_model(self, save_path=None):
        if self.filename is None and save_path is None:
            raise AttributeError
        elif save_path is None:
            save_path = str(self.log_path)

        torch.save(self.actor.state_dict(), save_path + "_actor_model.pt")
        torch.save(self.critic.state_dict(), save_path + "_critic_model.pt")

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
            [np.array]: processed action
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

        return np.mean(ep_ret)


if __name__ == "__main__":
    model = A2C()
    model.train()
