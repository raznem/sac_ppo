import torch
from torch.utils.data import Dataset

from rltoolkit.buffer import Memory


class AdvantageDataset(Dataset):
    def __init__(self, advantages: torch.tensor, buffer: Memory, normalize_adv: bool):
        if normalize_adv:
            advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1.2e-7
            )
        self.advantages = advantages
        self.action_logprobs = torch.cat(buffer.action_logprobs)
        self.actions = torch.cat(buffer.actions)
        self.obs = torch.cat(buffer.obs)

    def __len__(self):
        return len(self.advantages)

    def __getitem__(self, idx):
        adv = self.advantages[idx]
        act_logprob = self.action_logprobs[idx]
        act = self.actions[idx]
        obs = self.obs[idx]

        return adv, act_logprob, act, obs
