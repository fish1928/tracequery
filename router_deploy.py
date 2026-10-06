import json
import os

import torch
import torch.nn.functional as F

from router_llada import (
    FactoryRouter,
    FeatureBase,
    percentile_rank_masked,
    sanitize,
)

NEG_INF = torch.finfo(torch.float32).min

FEATURES_ATTENTION = {'attn_last', 'attn_all'}


def feature_dim(name, num_layers):
    return {
        'attn_last': 1,
        'attn_all': num_layers,
        'pos_delta': 2,
        'mask_density': 1,
        'conf': 1,
        'margin': 1,
        'conf_age': 2,
        'margin_age': 2,
        'conf_fresh': 2,
        'margin_fresh': 2,
    }[name]


def spec_dim_in(spec):
    return sum(feature_dim(name, spec['num_layers']) for name in spec['features'])


class Feature_deploy_stub(FeatureBase):
    def __init__(self, dim_in):
        super().__init__(folder_data=None)
        self._dim = int(dim_in)

    def dim(self):
        return self._dim

    def load_block(self, id_sample, pos_base, size_block):
        raise RuntimeError('deploy stub cannot load stats blocks; use build_online_x')


def save_router_bundle(router, spec, path_pt):
    router.save_checkpoint(path_pt)
    with open(path_pt[:-3] + '.json', 'w') as file:
        json.dump(spec, file, indent=2)


def load_router_bundle(path_pt, device='cpu'):
    with open(path_pt[:-3] + '.json', 'r') as file:
        spec = json.load(file)

    router = FactoryRouter.create(
        spec['router_name'],
        **spec.get('router_kwargs', {}),
    ).register_features(Feature_deploy_stub(spec['dim_in']))

    router.load_checkpoint(path_pt)
    router = router.to(device).eval()
    return router, spec


def masked_softmax_row(x, cand_3d, temperature):
    out = torch.softmax((x / temperature).masked_fill(~cand_3d, NEG_INF), dim=1)
    return torch.nan_to_num(out, nan=0.0).masked_fill(~cand_3d, 0.0)


def build_online_x(spec, attn_rows_all, conf_block, mask_still, pos_last, margin_block=None,
                   age_block=None):
    L = mask_still.shape[0]
    device = mask_still.device
    cand = mask_still.view(1, L)
    cand_3d = cand.unsqueeze(-1)

    temperature = float(spec.get('softmax_temperature', 1.0))
    window = int(spec.get('mask_density_window', 3))

    parts = []
    for name in spec['features']:
        if name in ('conf_age', 'margin_age'):
            assert age_block is not None, \
                f'spec wants {name} but the runner passed no age_block (pass snapshot.age)'
            value_block = conf_block if name == 'conf_age' else margin_block
            assert value_block is not None, \
                f'spec wants {name} but the runner passed no value table for it'
            max_age = float(spec.get('max_conf_age', 16))
            value_ranked = percentile_rank_masked(
                sanitize(value_block.view(1, L, 1)), cand)
            age_raw = (age_block.view(1, L, 1) / max(max_age, 1.0)).clamp(0.0, 1.0)
            age_raw = sanitize(age_raw).masked_fill(~cand_3d, 0.0)
            parts.append(torch.cat([value_ranked, age_raw], dim=-1))
            continue

        if name in ('conf_fresh', 'margin_fresh'):
            assert age_block is not None, \
                f'spec wants {name} but the runner passed no age_block (pass snapshot.age)'
            value_block = conf_block if name == 'conf_fresh' else margin_block
            assert value_block is not None, \
                f'spec wants {name} but the runner passed no value table for it'
            fresh_max_age = float(spec.get('fresh_max_age', 1))
            fresh_cand = cand & (age_block.view(1, L) <= fresh_max_age)
            value_ranked = percentile_rank_masked(
                sanitize(value_block.view(1, L, 1)), fresh_cand)
            parts.append(torch.cat([value_ranked, fresh_cand.float().unsqueeze(-1)], dim=-1))
            continue

        if name == 'attn_last':
            raw = attn_rows_all[-1].view(1, L, 1)
        elif name == 'attn_all':
            raw = attn_rows_all.t().contiguous().view(1, L, -1)
        elif name == 'conf':
            raw = conf_block.view(1, L, 1)
        elif name == 'margin':
            assert margin_block is not None, \
                'spec wants margin but the runner passed no margin_block ' \
                '(pass snapshot.margin for the current block)'
            raw = margin_block.view(1, L, 1)
        elif name == 'pos_delta':
            positions = torch.arange(L, device=device)
            delta = (positions - pos_last).float() / L
            raw = torch.stack([delta, delta.abs()], dim=-1).view(1, L, 2)
        elif name == 'mask_density':
            kernel = torch.ones(1, 1, 2 * window + 1, device=device)
            count = F.conv1d(mask_still.float().view(1, 1, L), kernel, padding=window)
            count = torch.round(count)
            raw = (count / (2 * window + 1)).view(1, L, 1)
        else:
            raise ValueError(f'unknown online feature: {name}')

        raw = sanitize(raw)

        if spec['normalization'] == 'softmax_attn' and name in FEATURES_ATTENTION:
            parts.append(masked_softmax_row(raw, cand_3d.expand_as(raw), temperature))
        else:
            parts.append(percentile_rank_masked(raw, cand))

    return torch.cat(parts, dim=-1)


@torch.no_grad()
def select_topk_candidates(router, spec, attn_rows_all, conf_block, mask_still, pos_last, k,
                           margin_block=None, age_block=None):
    x = build_online_x(spec, attn_rows_all, conf_block, mask_still, pos_last, margin_block,
                       age_block)
    scores = router(x).view(-1)
    scores = scores.masked_fill(~mask_still, NEG_INF)
    return scores.topk(k).indices
