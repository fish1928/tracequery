import itertools
import json
import os

import torch

from router_llada import (
    FactoryLoss,
    FactoryRouter,
    Feature_attn_all,
    Feature_attn_last,
    Feature_conf,
    Feature_mask_density,
    Feature_pos_delta,
    Feature_rank_normed,
    Feature_softmax_row,
    FeatureBase,
    RouterTrainer,
    build_geometry,
    load_stat,
    sanitize,
)
from router_deploy import FEATURES_ATTENTION, save_router_bundle, spec_dim_in
from tools_debug import jprint


FOLDER_DATA_GSM8K = os.environ['FOLDER_DATA_GSM8K']
FOLDER_DATA_IFEVAL = os.environ['FOLDER_DATA_IFEVAL']
DEVICE = os.environ.get('DEVICE', 'cuda:0')
FOLDER_OUT = os.environ.get('FOLDER_OUT', 'routers_e2e')
GRID_FILTER = os.environ.get('GRID_FILTER', '')

NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', 32))
SIZE_BLOCK_GSM8K = int(os.environ.get('SIZE_BLOCK_GSM8K', 128))
SIZE_BLOCK_IFEVAL = int(os.environ.get('SIZE_BLOCK_IFEVAL', 256))

HORIZON = 5
MAX_CONF_AGE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HOLDOUT = 0.2
FILTER_RESULT = 'all'
SEED = 233

ROUTER_NAME = 'mlp'
ROUTER_KWARGS = {'dim_hidden': 64, 'num_blocks_mlp': 2}

FEATURE_SETS = {
    'attn_last_geo_conf': ['attn_last', 'pos_delta', 'mask_density', 'conf'],
    'attn_last_geo': ['attn_last', 'pos_delta', 'mask_density'],
    'attn_all': ['attn_all'],
}

DATASETS = {
    'gsm8k': [(FOLDER_DATA_GSM8K, SIZE_BLOCK_GSM8K)],
    'ifeval': [(FOLDER_DATA_IFEVAL, SIZE_BLOCK_IFEVAL)],
    'mix': [(FOLDER_DATA_GSM8K, SIZE_BLOCK_GSM8K), (FOLDER_DATA_IFEVAL, SIZE_BLOCK_IFEVAL)],
}

NORMALIZATIONS = ['rank', 'softmax_attn']
LOSSES = ['decay_within_h', 'plackett_luce']


class Feature_conf_random_aged(FeatureBase):
    def __init__(self, folder_data, max_age=MAX_CONF_AGE):
        super().__init__(folder_data)
        self.max_age = int(max_age)

    def dim(self):
        return 1

    def load_block(self, id_sample, pos_base, size_block):
        conf = sanitize(load_stat(self._folder_base(id_sample), 'conf', pos_base, size_block))
        T = conf.shape[0]

        max_age_per_row = torch.arange(T).clamp(max=self.max_age)
        ages = torch.floor(torch.rand(T, conf.shape[1]) * (max_age_per_row[:, None].float() + 1.0)).long()
        source_row = (torch.arange(T)[:, None] - ages).clamp(min=0)

        return conf.gather(dim=0, index=source_row).unsqueeze(-1)


def make_offline_features(spec, folder_data):
    features = []
    for name in spec['features']:
        if name == 'attn_last':
            base = Feature_attn_last(folder_data)
        elif name == 'attn_all':
            base = Feature_attn_all(folder_data, num_layers=spec['num_layers'])
        elif name == 'pos_delta':
            base = Feature_pos_delta(folder_data)
        elif name == 'mask_density':
            base = Feature_mask_density(folder_data, window=spec['mask_density_window'])
        elif name == 'conf':
            if spec['conf_mode'] == 'fresh':
                base = Feature_conf(folder_data)
            else:
                base = Feature_conf_random_aged(folder_data, max_age=MAX_CONF_AGE)
        else:
            raise ValueError(name)

        if spec['normalization'] == 'softmax_attn' and name in FEATURES_ATTENTION:
            features.append(Feature_softmax_row(base, temperature=spec['softmax_temperature']))
        else:
            features.append(Feature_rank_normed(base))
    return features


def train_version(name_version, spec):
    torch.manual_seed(SEED)

    datasets = DATASETS[spec['dataset']]

    trainers = [
        RouterTrainer(
            folder_data, h=HORIZON, size_block=size_block, device=DEVICE,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, holdout=HOLDOUT,
            filter_result=FILTER_RESULT, seed=SEED,
        )
        for folder_data, size_block in datasets
    ]

    feature_lists = [make_offline_features(spec, folder_data) for folder_data, _ in datasets]

    router = FactoryRouter.create(ROUTER_NAME, **ROUTER_KWARGS)
    router.register_features(*feature_lists[0])
    assert router.dim_in == spec['dim_in']
    router = router.to(DEVICE)
    for trainer in trainers:
        trainer.router = router

    loss = FactoryLoss.create(spec['loss'])
    optimizer = torch.optim.AdamW(router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    router.train()
    for id_epoch in range(NUM_EPOCHS):
        losses = []
        for trainer, features in zip(trainers, feature_lists):
            router.features = features
            for x, order in trainer._iter_blocks(trainer.ids_train):
                gap, cand_mask = build_geometry(order.cpu(), trainer.size_block)
                gap, cand_mask = gap.to(DEVICE), cand_mask.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                loss_value = loss(router(x), gap, cand_mask, HORIZON)
                loss_value.backward()
                optimizer.step()
                losses.append(float(loss_value.item()))
        jprint(f'[{name_version}] epoch {id_epoch}: loss {sum(losses) / len(losses):.4f} over {len(losses)} blocks')

    router.eval()
    names_dataset = ['gsm8k', 'ifeval'] if spec['dataset'] == 'mix' else [spec['dataset']]
    recalls = {}
    for name_dataset, trainer, features in zip(names_dataset, trainers, feature_lists):
        router.features = features
        recalls[name_dataset] = trainer.evaluate(hs=[HORIZON])[f'recall@{HORIZON}']
    spec['recall_holdout'] = recalls

    router.features = feature_lists[0]
    os.makedirs(FOLDER_OUT, exist_ok=True)
    path_pt = os.path.join(FOLDER_OUT, name_version + '.pt')
    save_router_bundle(router, spec, path_pt)
    jprint(f'[{name_version}] saved {path_pt} recall_holdout={recalls}')


def build_grid():
    for name_fs, names_feature in FEATURE_SETS.items():
        conf_modes = ['fresh', 'aged'] if 'conf' in names_feature else ['none']
        for conf_mode, norm, name_loss, name_dataset in itertools.product(
                conf_modes, NORMALIZATIONS, LOSSES, DATASETS.keys()):

            name_version = f'feat-{name_fs}__conf-{conf_mode}__norm-{norm}__loss-{name_loss}__data-{name_dataset}'
            spec = {
                'features': list(names_feature),
                'conf_mode': conf_mode,
                'normalization': norm,
                'softmax_temperature': 1.0,
                'mask_density_window': 3,
                'loss': name_loss,
                'dataset': name_dataset,
                'router_name': ROUTER_NAME,
                'router_kwargs': dict(ROUTER_KWARGS),
                'num_layers': NUM_LAYERS,
                'h': HORIZON,
                'max_conf_age': MAX_CONF_AGE,
                'seed': SEED,
                'num_epochs': NUM_EPOCHS,
            }
            spec['dim_in'] = spec_dim_in(spec)
            yield name_version, spec


if __name__ == '__main__':
    versions = [(n, s) for n, s in build_grid() if GRID_FILTER in n]
    jprint(f'training {len(versions)} router versions -> {FOLDER_OUT}')

    for name_version, spec in versions:
        try:
            train_version(name_version, spec)
        except Exception as error:
            jprint(f'[{name_version}] FAILED: {error}')
