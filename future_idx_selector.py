import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os

from attn_order_eval import ScoreOrderEval, summ

MYDTYPE = torch.float64
MYDEVICE = 'cuda:0'


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


@dataclass
class SimpleMLPConfig:
    dim_hidden: int
    dim_in: int
    dim_out: int
    bias: bool
    activation: nn.Module

@dataclass
class SimpleTrainConfig:
    T: int
    L: int
    folder_root: str


class SimpleMLP(nn.Module):
    def __init__(self, config_mlp):
        super().__init__()

        self.dim_hidden = config_mlp.dim_hidden
        self.dim_in = config_mlp.dim_in
        self.dim_out = config_mlp.dim_out
        self.bias = config_mlp.bias

        self.project_gate = nn.Linear(self.dim_in, self.dim_hidden, bias=self.bias, dtype=MYDTYPE)
        self.project_up = nn.Linear(self.dim_in, self.dim_hidden, bias=self.bias, dtype=MYDTYPE)
        self.project_down = nn.Linear(self.dim_hidden, self.dim_out, bias=self.bias, dtype=MYDTYPE)
        self.activation = config_mlp.activation

    def forward(self, x):
        return self.project_down(self.activation(self.project_gate(x)) * self.project_up(x))

    def device(self):
        return next(self.parameters()).device


class RandomModel:
    def __call__(self, x, *args, **kwargs):
        x = x[:,:,-1]
        return torch.rand(x.shape, device=x.device).unsqueeze(-1)


class FutureIDXSelector:
    def __init__(self, model, select_only_in_h=True, h=5):
        self.model = model
        self.select_only_in_h = select_only_in_h
        self.h = h

    def select_future_by_attn(self, attn):
        index_avail = (attn >0).nonzero(as_tuple=True)[1].reshape(attn.shape[0], -1)
        attn_avail = torch.gather(attn, -1, index_avail)
        scores = self.model(attn_avail.unsqueeze(-1)).squeeze(-1)
        idx = scores.argsort(dim=-1, descending=True)[:, :self.h]
        return torch.gather(index_avail, 1, idx)

    def select_future_by_attn_full(self, attn):
        index_avail = (attn >0).nonzero(as_tuple=True)[1].reshape(attn.shape[0], -1, attn.shape[-1])
        attn_avail = torch.gather(attn, 1, index_avail)
        scores = self.model(attn_avail).squeeze(-1)

        idx = scores.argsort(dim=-1, descending=True)[:, :self.h]
        return torch.gather(index_avail[:,:,0], 1, idx)

    def select_future_by_3(self, met):
        attn = met[:, :, -1]
        index_avail = (attn >0).nonzero(as_tuple=True)[1].reshape(attn.shape[0], -1)
        index_avail_3 = index_avail.view(1,-1,1).expand(1,-1,3)
        met_avail = torch.gather(met, 1, index_avail_3)
        scores = self.model(met_avail).squeeze(-1)
        
        idx = scores.argsort(dim=-1, descending=True)[:, :self.h]
        return torch.gather(index_avail, 1, idx)

def generate_mask_sequence(unmask):
    L, T = unmask.shape[0], unmask.shape[0]
    a = torch.full((L,), -1, dtype=torch.long)
    a[unmask] = torch.arange(T)
    b = a.view(1, -1) - torch.arange(T).view(-1, 1)
    return b

def generate_y(unmask, l, h=5):
    a = torch.ones(l, dtype=torch.long, device=MYDEVICE)
    a[unmask] = torch.arange(l, device=MYDEVICE)
    b = a.view(1,-1) - torch.arange(l, device=MYDEVICE).view(-1,1)
    mask_current = (b <= h) & (b > 0)
    b[mask_current] = (h+1 - b[mask_current])
    b[~mask_current] = 0
    b = b/b[0].sum()

    return b

def soft_rank_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(pred, dim=-1)
    loss = -(y * log_probs).sum(dim=-1)
    return loss.mean()


class FutureIdxSelectorModelLoader:
    def __init__(self, dim_in, device):
        config_mlp1 = SimpleMLPConfig(
            dim_in=dim_in,
            dim_hidden=64,
            dim_out=64,
            bias=True,
            activation=SiLU()
        )

        config_mlp3 = SimpleMLPConfig(
            dim_in=64,
            dim_hidden=64,
            dim_out=1,
            bias=True,
            activation=SiLU()
        )

        self.model = nn.Sequential(
            SimpleMLP(config_mlp1),
            SimpleMLP(config_mlp3)
        )
        self.device = device

    def load(self, path_model):
        state = torch.load(path_model, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        assert len(missing) == 0, f'missing keys: {missing}'
        assert len(unexpected) == 0, f'unpexted keys: {unexpected}'
        self.model.to(self.device)
        self.model.eval()
        return self.model
