import os

if os.environ.get("JINYU_DEBUG", False):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time

from tools_llada import RunnerReport
from plugins_llada import SaveKVPreviousPlugin_Disabled, CachePastKVPlugin_Enabled,\
                            CacheAttnPlugin_Enabled, CacheVOPlugin_Disabled

from future_idx_selector import FutureIDXSelector, FutureIdxSelectorModelLoader

from router_deploy import load_router_bundle

from tools_debug import jprint

from constants_llada import DTYPE_EVAL, NAME_MLP


class RunModelMLPBase:

    def __init__(self):
        self.mlp = None
        self.router_bundle = None
        self.report = RunnerReport()
        self.ids_stop = None

    def config_plugin_(self, config):
        config.klass_save_kv_previous=SaveKVPreviousPlugin_Disabled
        config.klass_cache_past_kv=CachePastKVPlugin_Enabled
        config.klass_cache_attn=CacheAttnPlugin_Enabled
        config.klass_cache_vo=CacheVOPlugin_Disabled

        return self

    def register_plugin_(self, model, config):
        model\
            .fill_plugin(config.klass_cache_past_kv)\
            .fill_plugin(config.klass_save_kv_previous)\
            .fill_plugin(config.klass_cache_attn)\
            .fill_plugin(config.klass_cache_vo)

    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        raise NotImplementedError('thread runner must implement generate()')

    def run_one(self, model, tokenizer, config, *args, **kwargs):

        config.klass_cache_attn.set_size_block(config.size_block)
        config.klass_cache_attn.set_len_prompt(kwargs['len_prompt'])

        path_router = getattr(config, 'path_router', None)
        if path_router:
            if self.router_bundle is None:
                router, spec_router = load_router_bundle(path_router, device=config.device)
                self.router_bundle = (router, spec_router)
                jprint(f'loaded router bundle {path_router}: {spec_router["features"]} norm={spec_router["normalization"]}')
        elif self.mlp is None:
            loader_mlp = FutureIdxSelectorModelLoader(1, config.device)
            self.mlp = loader_mlp.load(NAME_MLP).to(DTYPE_EVAL)

        kwargs_selector = {}
        if config.h:
            kwargs_selector['h'] = config.h

        if config.select_only_in_h is not None:
            kwargs_selector['select_only_in_h'] = bool(config.select_only_in_h)

        future_idx_selector = FutureIDXSelector(self.mlp, **kwargs_selector)

        plugin_cache_past_kv = config.klass_cache_past_kv()
        plugin_cache_attn = config.klass_cache_attn()

        plugin_cache_past_kv.clear(model)
        plugin_cache_attn.clear(model)

        kwargs['future_idx_selector'] = future_idx_selector
        kwargs['plugin_cache_attn'] = plugin_cache_attn
        kwargs['router_bundle'] = self.router_bundle

        time_start = time.perf_counter()
        sentence_generated, has_done = self.generate(
            model,
            tokenizer,
            config,
            *args,
            **kwargs
        )
        duration_s = time.perf_counter() - time_start

        self.report.add_and_dump(config, kwargs['len_prompt'], has_done, duration_s)

        return sentence_generated, has_done
