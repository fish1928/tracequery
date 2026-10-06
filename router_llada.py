import json
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from attn_order_eval import ScoreOrderEval, summ
from tools_debug import jprint

NEG_INF = torch.finfo(torch.float32).min


def load_stat(folder_base, name, pos_base, size_block):
    path = os.path.join(folder_base, f'{name}_{pos_base}_{pos_base + size_block}.pt')
    return torch.load(path, map_location='cpu')


def sanitize(value):
    value = value.float()
    bad = ~torch.isfinite(value) | (value <= -1e30)
    return torch.where(bad, torch.zeros_like(value), value)


def build_geometry(order, size_block):
    T = order.shape[0]
    step_of = torch.full((size_block,), -1, dtype=torch.long)
    step_of[order] = torch.arange(T)

    gap = step_of.view(1, -1) - torch.arange(T).view(T, 1)
    cand_mask = gap > 0
    return gap, cand_mask


def percentile_rank_masked(values, mask):
    squeeze_last = values.ndim == 2
    x = values.unsqueeze(-1) if squeeze_last else values
    m = mask.unsqueeze(-1).expand_as(x)

    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)
    m_j = m.unsqueeze(1)

    num_lower = ((x_j < x_i) & m_j).sum(dim=2)
    num_equal = ((x_j == x_i) & m_j).sum(dim=2)
    average_rank = num_lower.float() + 0.5 * (num_equal.float() - 1.0)

    num_valid = m.sum(dim=1, keepdim=True).float()
    out = average_rank / (num_valid - 1.0).clamp(min=1.0)
    out = out.masked_fill(~m, 0.0)

    return out.squeeze(-1) if squeeze_last else out


class FeatureBase(ABC):

    def __init__(self, folder_data):
        self.folder_data = folder_data

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def dim(self):
        raise NotImplementedError

    @abstractmethod
    def load_block(self, id_sample, pos_base, size_block):
        raise NotImplementedError

    def _folder_base(self, id_sample):
        return os.path.join(self.folder_data, str(id_sample))

    def _order_local(self, id_sample, pos_base, size_block):
        unmask = load_stat(self._folder_base(id_sample), 'unmask', pos_base, size_block)
        return unmask.squeeze(-1).long() - pos_base


class Feature_conf(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'conf', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)


class Feature_margin(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'margin', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)


class Feature_entropy(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        value = load_stat(self._folder_base(id_sample), 'entropy', pos_base, size_block)
        return sanitize(value).unsqueeze(-1)


class Feature_attn_last(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        attn = load_stat(self._folder_base(id_sample), 'attn', pos_base, size_block)
        return sanitize(attn.squeeze(-2)[:, -1, :]).unsqueeze(-1)


class Feature_attn_all(FeatureBase):
    def __init__(self, folder_data, num_layers=32):
        super().__init__(folder_data)
        self.num_layers = num_layers

    def dim(self):
        return self.num_layers

    def load_block(self, id_sample, pos_base, size_block):
        attn = load_stat(self._folder_base(id_sample), 'attn', pos_base, size_block)
        attn = attn.squeeze(-2)
        assert attn.shape[1] == self.num_layers,\
            f'num_layers mismatch: file has {attn.shape[1]}, feature configured {self.num_layers}'
        return sanitize(attn).permute(0, 2, 1)


class Feature_pos_delta(FeatureBase):
    def dim(self):
        return 2

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        positions = torch.arange(size_block).view(1, -1)
        delta = (positions - order.view(-1, 1)).float() / size_block
        return torch.stack([delta, delta.abs()], dim=-1)


class Feature_step_progress(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        T = order.shape[0]
        progress = (torch.arange(T).float() / max(T - 1, 1)).view(-1, 1, 1)
        return progress.expand(T, size_block, 1).clone()


class Feature_mask_density(FeatureBase):
    def __init__(self, folder_data, window=3):
        super().__init__(folder_data)
        self.window = window

    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        gap, cand_mask = build_geometry(order, size_block)
        masked = cand_mask.float()

        kernel = torch.ones(1, 1, 2 * self.window + 1)
        count = F.conv1d(masked.unsqueeze(1), kernel, padding=self.window).squeeze(1)
        count = torch.round(count)
        density = count / (2 * self.window + 1)
        return density.unsqueeze(-1)


class Feature_x0_stability(FeatureBase):
    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        x0 = load_stat(self._folder_base(id_sample), 'x0', pos_base, size_block)
        stable = torch.zeros(x0.shape, dtype=torch.float32)
        stable[1:] = (x0[1:] == x0[:-1]).float()
        return stable.unsqueeze(-1)


class FeatureWrapperBase(FeatureBase):

    def __init__(self, feature_inner):
        super().__init__(feature_inner.folder_data)
        self.feature_inner = feature_inner

    def dim(self):
        return self.feature_inner.dim()

    def get_name(self):
        return f'{self._tag()}({self.feature_inner.get_name()})'

    def _tag(self):
        return self.__class__.__name__.replace('Feature_', '')

    def _cand_mask_3d(self, id_sample, pos_base, size_block):
        order = self._order_local(id_sample, pos_base, size_block)
        _, cand_mask = build_geometry(order, size_block)
        return cand_mask.unsqueeze(-1)


class Feature_rank_normed(FeatureWrapperBase):

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand_mask = self._cand_mask_3d(id_sample, pos_base, size_block).squeeze(-1)
        return percentile_rank_masked(x, cand_mask)


class Feature_znormed_row(FeatureWrapperBase):
    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x).float()

        n = cand.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * cand).sum(dim=1, keepdim=True) / n
        var = ((x - mean).pow(2) * cand).sum(dim=1, keepdim=True) / n
        out = (x - mean) / (var.sqrt() + 1e-6)
        return out * cand


class Feature_minmax_row(FeatureWrapperBase):
    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)

        x_min = x.masked_fill(~cand, float('inf')).amin(dim=1, keepdim=True)
        x_max = x.masked_fill(~cand, float('-inf')).amax(dim=1, keepdim=True)
        out = (x - x_min) / (x_max - x_min + 1e-6)
        return sanitize(out).masked_fill(~cand, 0.0)


class Feature_softmax_row(FeatureWrapperBase):
    def __init__(self, feature_inner, temperature=1.0):
        super().__init__(feature_inner)
        self.temperature = temperature

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)

        out = torch.softmax((x / self.temperature).masked_fill(~cand, NEG_INF), dim=1)
        return torch.nan_to_num(out, nan=0.0).masked_fill(~cand, 0.0)


class Feature_log_scaled(FeatureWrapperBase):
    def __init__(self, feature_inner, eps=1e-6):
        super().__init__(feature_inner)
        self.eps = eps

    def load_block(self, id_sample, pos_base, size_block):
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
        return torch.log(x.clamp(min=0.0) + self.eps).masked_fill(~cand, 0.0)


class Feature_znormed_global(FeatureWrapperBase):
    def __init__(self, feature_inner):
        super().__init__(feature_inner)
        self.mean = None
        self.std = None

    def fitted(self):
        return self.mean is not None

    def fit(self, blocks, size_block):
        xs = []
        for id_sample, pos_base in blocks:
            x = self.feature_inner.load_block(id_sample, pos_base, size_block)
            cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
            xs.append(x[cand].reshape(-1, self.dim()))
        x_all = torch.cat(xs, dim=0)
        self.mean = x_all.mean(dim=0)
        self.std = x_all.std(dim=0) + 1e-6
        return self

    def get_state(self):
        return {'mean': self.mean, 'std': self.std}

    def set_state(self, state):
        self.mean, self.std = state['mean'], state['std']

    def load_block(self, id_sample, pos_base, size_block):
        assert self.fitted(), 'Feature_znormed_global must be fitted (the trainer fits it on register_router)'
        x = self.feature_inner.load_block(id_sample, pos_base, size_block)
        cand = self._cand_mask_3d(id_sample, pos_base, size_block).expand_as(x)
        return ((x - self.mean) / self.std).masked_fill(~cand, 0.0)


class RouterBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = []
        self.dim_in = 0

    def register_features(self, *features):
        self.features = list(features)
        self.dim_in = sum(f.dim() for f in features)
        self._build()
        return self

    def _build(self):
        pass

    def trainable(self):
        return any(p.requires_grad for p in self.parameters())

    def build_block_x(self, id_sample, pos_base, size_block):
        assert self.features, 'call register_features first'
        return torch.cat([f.load_block(id_sample, pos_base, size_block) for f in self.features], dim=-1)

    def forward(self, x):
        raise NotImplementedError

    def describe(self):
        return {
            'router': self.__class__.__name__,
            'features': [f.get_name() for f in self.features],
            'dim_in': self.dim_in,
        }

    def save_checkpoint(self, path):
        features_state = [f.get_state() if hasattr(f, 'get_state') else None for f in self.features]
        torch.save({
            'state_dict': self.state_dict(),
            'describe': self.describe(),
            'features_state': features_state,
        }, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(state['state_dict'])

        for feature, feature_state in zip(self.features, state.get('features_state', [])):
            if feature_state is not None and hasattr(feature, 'set_state'):
                feature.set_state(feature_state)
        return self


class Router_linear(RouterBase):
    def _build(self):
        self.head = nn.Linear(self.dim_in, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class GatedMLPBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, bias=True):
        super().__init__()
        self.project_gate = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.project_up = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.project_down = nn.Linear(dim_hidden, dim_out, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.project_down(self.activation(self.project_gate(x)) * self.project_up(x))


class Router_mlp(RouterBase):
    def __init__(self, dim_hidden=64, num_blocks_mlp=2):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_blocks_mlp = num_blocks_mlp

    def _build(self):
        blocks = []
        dim_current = self.dim_in
        for _ in range(self.num_blocks_mlp - 1):
            blocks.append(GatedMLPBlock(dim_current, self.dim_hidden, self.dim_hidden))
            dim_current = self.dim_hidden
        blocks.append(GatedMLPBlock(dim_current, self.dim_hidden, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x).squeeze(-1)


class Router_set_attention(RouterBase):
    def __init__(self, dim_model=32, num_heads=1, dim_hidden=64):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden

    def _build(self):
        self.embed = nn.Linear(self.dim_in, self.dim_model)
        self.attention = nn.MultiheadAttention(self.dim_model, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.dim_model)
        self.head = GatedMLPBlock(self.dim_model, self.dim_hidden, 1)

    def forward(self, x):
        e = self.embed(x)
        a, _ = self.attention(e, e, e, need_weights=False)
        e = self.norm(e + a)
        return self.head(e).squeeze(-1)


class Router_mockup_random(RouterBase):
    def __init__(self, seed=233):
        super().__init__()
        self.generator = torch.Generator().manual_seed(seed)

    def trainable(self):
        return False

    def forward(self, x):
        return torch.rand(x.shape[:-1], generator=self.generator, device='cpu').to(x.device)


class Router_mockup_raw_feature(RouterBase):
    def trainable(self):
        return False

    def forward(self, x):
        return x[..., 0]


class Router_mockup_nearest_right(RouterBase):
    def trainable(self):
        return False

    def forward(self, x):
        delta = x[..., 0]
        return torch.where(delta > 0, -delta, -(2.0 + delta.abs()))


class LossBase(ABC):
    @abstractmethod
    def __call__(self, scores, gap, cand_mask, h):
        raise NotImplementedError


def _masked_log_softmax(scores, mask):
    return F.log_softmax(scores.masked_fill(~mask, NEG_INF), dim=-1)


def _soft_ce(scores, cand_mask, y):
    logp = _masked_log_softmax(scores, cand_mask)
    loss_rows = -torch.where(y > 0, y * logp, torch.zeros_like(y)).sum(-1)
    row_valid = y.sum(-1) > 0
    assert bool(row_valid.any()), 'no row has a positive target'
    return loss_rows[row_valid].mean()


class Loss_uniform_within_h(LossBase):
    def __call__(self, scores, gap, cand_mask, h):
        pos = (gap >= 1) & (gap <= h)
        y = pos.float()
        y = y / y.sum(-1, keepdim=True).clamp(min=1.0)
        return _soft_ce(scores, cand_mask, y)


class Loss_decay_within_h(LossBase):
    def __call__(self, scores, gap, cand_mask, h):
        pos = (gap >= 1) & (gap <= h)
        y = (h + 1 - gap).float() * pos.float()
        y = y / y.sum(-1, keepdim=True).clamp(min=1.0)
        return _soft_ce(scores, cand_mask, y)


class Loss_bce_within_h(LossBase):
    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight

    def __call__(self, scores, gap, cand_mask, h):
        pos = ((gap >= 1) & (gap <= h)).float()
        scores_flat = scores[cand_mask]
        target_flat = pos[cand_mask]

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=scores.device)
        return F.binary_cross_entropy_with_logits(scores_flat, target_flat, pos_weight=pos_weight)


class Loss_plackett_luce(LossBase):
    def __call__(self, scores, gap, cand_mask, h):
        losses = []
        for j in range(1, h + 1):
            remaining = cand_mask & (gap >= j)
            target = gap == j
            row_valid = target.any(-1) & (remaining.sum(-1) > 1)
            if not bool(row_valid.any()):
                continue
            logp = _masked_log_softmax(scores, remaining)
            ll = torch.where(target, logp, torch.zeros_like(logp)).sum(-1)
            losses.append(-ll[row_valid])
        assert losses, 'no valid (row, j) pairs'
        return torch.cat(losses).mean()


class FactoryFeature:
    _REGISTRY = {
        'conf': Feature_conf,
        'margin': Feature_margin,
        'entropy': Feature_entropy,
        'attn_last': Feature_attn_last,
        'attn_all': Feature_attn_all,
        'pos_delta': Feature_pos_delta,
        'step_progress': Feature_step_progress,
        'mask_density': Feature_mask_density,
        'x0_stability': Feature_x0_stability,
    }

    _REGISTRY_WRAPPER = {
        'rank': Feature_rank_normed,
        'znorm_row': Feature_znormed_row,
        'minmax_row': Feature_minmax_row,
        'softmax_row': Feature_softmax_row,
        'log': Feature_log_scaled,
        'znorm_global': Feature_znormed_global,
    }

    @classmethod
    def create(cls, name, folder_data, **kwargs):
        return cls._REGISTRY[name](folder_data, **kwargs)

    @classmethod
    def wrap(cls, name, feature_inner, **kwargs):
        return cls._REGISTRY_WRAPPER[name](feature_inner, **kwargs)


class FactoryRouter:
    _REGISTRY = {
        'linear': Router_linear,
        'mlp': Router_mlp,
        'set_attention': Router_set_attention,
        'mockup_random': Router_mockup_random,
        'mockup_raw': Router_mockup_raw_feature,
        'mockup_nearest_right': Router_mockup_nearest_right,
    }

    @classmethod
    def create(cls, name, **kwargs):
        return cls._REGISTRY[name](**kwargs)


class FactoryLoss:
    _REGISTRY = {
        'uniform_within_h': Loss_uniform_within_h,
        'decay_within_h': Loss_decay_within_h,
        'bce_within_h': Loss_bce_within_h,
        'plackett_luce': Loss_plackett_luce,
    }

    @classmethod
    def create(cls, name, **kwargs):
        return cls._REGISTRY[name](**kwargs)


class RouterTrainer:

    def __init__(self, folder_data, h=5, size_block=64, device='cpu',
                 lr=1e-3, weight_decay=1e-4, holdout=0.2, filter_result='all', seed=233):
        self.folder_data = folder_data
        self.h = h
        self.size_block = size_block
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.holdout = holdout
        self.filter_result = filter_result
        self.seed = seed

        self.router = None
        self.loss = None

        self.ids_train, self.ids_eval = self._split_samples()

    def register_router(self, router):
        self.router = router.to(self.device)

        for feature in router.features:
            if hasattr(feature, 'fit') and hasattr(feature, 'fitted') and not feature.fitted():
                feature.fit(list(self._list_blocks(self.ids_train)), self.size_block)
                jprint(f'fitted {feature.get_name()} on {len(self.ids_train)} train samples')
        return self

    def register_loss(self, loss):
        self.loss = loss() if isinstance(loss, type) else loss
        return self

    def _read_result(self, id_sample):
        path = os.path.join(self.folder_data, str(id_sample), 'generated.json')
        if not os.path.exists(path):
            return 'unknown'
        with open(path, 'r') as file:
            return json.load(file).get('result', 'unknown')

    def _split_samples(self):
        ids_all = sorted(int(f) for f in os.listdir(self.folder_data) if f.isdigit())

        if self.filter_result in ('pass', 'fail'):
            ids_all = [i for i in ids_all if self._read_result(i) == self.filter_result]
        assert ids_all, f'no sample folders (filter_result={self.filter_result})'

        n_eval = max(1, int(len(ids_all) * self.holdout))
        return ids_all[:-n_eval], ids_all[-n_eval:]

    def _list_blocks(self, ids_sample):
        for id_sample in ids_sample:
            folder_base = os.path.join(self.folder_data, str(id_sample))
            with open(os.path.join(folder_base, '.pos_root'), 'r') as file:
                pos_root = int(file.read())
            num_blk = len([f for f in os.listdir(folder_base) if f.startswith('unmask_') and f.endswith('.pt')])

            for id_blk in range(num_blk):
                yield id_sample, pos_root + id_blk * self.size_block

    def _iter_blocks(self, ids_sample):
        for id_sample, pos_base in self._list_blocks(ids_sample):
            folder_base = os.path.join(self.folder_data, str(id_sample))
            x = self.router.build_block_x(id_sample, pos_base, self.size_block).to(self.device)
            unmask = load_stat(folder_base, 'unmask', pos_base, self.size_block)
            order = unmask.squeeze(-1).long() - pos_base
            yield x, order.to(self.device)

    def train(self, num_epochs=10, log_every=1):
        assert self.router is not None and self.loss is not None

        if not self.router.trainable():
            jprint(f'{self.router.__class__.__name__} is a mockup; nothing to train')
            return self

        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(self.router.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for id_epoch in range(num_epochs):
            losses = []
            for x, order in self._iter_blocks(self.ids_train):
                gap, cand_mask = build_geometry(order.cpu(), self.size_block)
                gap, cand_mask = gap.to(self.device), cand_mask.to(self.device)

                optimizer.zero_grad()
                scores = self.router(x)
                loss = self.loss(scores, gap, cand_mask, self.h)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if id_epoch % log_every == 0:
                jprint(f'epoch {id_epoch}: loss {sum(losses) / len(losses):.4f} over {len(losses)} blocks')
        return self

    @torch.no_grad()
    def evaluate(self, hs=None, ids_sample=None):
        assert self.router is not None
        hs = hs or [3, self.h, 2 * self.h]
        ids_sample = ids_sample if ids_sample is not None else self.ids_eval

        values = {h: [] for h in hs}
        values_ap = []
        values_ndgc = []
        for x, order in self._iter_blocks(ids_sample):
            scores = self.router(x)
            evaluator = ScoreOrderEval(scores.cpu(), order.cpu())
            for h in hs:
                values[h].append(evaluator.recall_at_h(h))
            values_ap.append(evaluator.pr_auc(self.h))
            values_ndgc.append(evaluator.ndcg_at_h(self.h))

        report = {f'recall@{h}': summ(torch.cat(values[h])) for h in hs}
        report[f'pr_auc@{self.h}'] = summ(torch.cat(values_ap))
        report[f'ndgc@{self.h}'] = summ(torch.cat(values_ndgc))
        report['n_blocks'] = len(values_ap)
        report['router'] = self.router.describe()
        return report
