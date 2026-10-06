import torch

from components_llada import SimpleLogitsSnapshot
from tools_llada import BlockDiffusionQuotaHelper
from router_deploy import select_topk_candidates
from runner_mlp_common import RunModelMLPBase
from plugins_llada import CachePastKVPlugin_Enabled


class RunModel(RunModelMLPBase):

    def generate(self, model, tokenizer, config_diffusion, *args, **kwargs):
        num_blocks = config_diffusion.num_blocks
        step_per_block = config_diffusion.step_per_block
        size_block = config_diffusion.size_block
        id_mask = config_diffusion.id_mask
        sorter = config_diffusion.klass_sorter()
        collector = config_diffusion.klass_collector()

        step_refresh_remainder = config_diffusion.step_refresh_remainder
        remainder_prompt = getattr(config_diffusion, 'step_refresh_remainder_prompt', None)

        words_stop = list(kwargs['until'])
        len_prompt = kwargs['len_prompt']
        x = kwargs['ids_input']
        mask_attention = kwargs.get('attention_mask')
        if mask_attention is None:
            mask_attention = torch.ones_like(x)

        plugin_cache_attn = kwargs['plugin_cache_attn']
        future_idx_selector = kwargs['future_idx_selector']
        router_bundle = kwargs.get('router_bundle')
        assert router_bundle is not None, \
            'the batch runner needs a router bundle (path_router); legacy MLP not ported'
        router, spec_router = router_bundle

        B = x.shape[0]
        device = x.device
        h = future_idx_selector.h
        has_done = [False] * B
        sentences = [''] * B

        position_start, position_end = 0, len_prompt
        idx_prompt = torch.arange(position_end, dtype=torch.long, device=device)
        shape_target = (B, position_end, -1)
        model(x[:, idx_prompt], idx_current=idx_prompt, shape_target=shape_target,
              skip_logits=True, attention_mask=mask_attention[:, :position_end])
        snapshot = SimpleLogitsSnapshot(x[:, idx_prompt], x[:, idx_prompt], id_mask)

        idx_transform_2d = None

        for id_block in range(num_blocks):
            position_start = len_prompt + id_block * size_block
            position_end = position_start + size_block
            mask_mask_block = x[:, position_start:position_end] == id_mask
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_block, step_per_block)

            if future_idx_selector.select_only_in_h:
                assert h >= quota_helper.get_quota_max(), \
                    'horizon h must cover the per-step unmask quota: {} >= {}'.format(
                        h, quota_helper.get_quota_max())

            idx_block = torch.arange(position_start, position_end, dtype=torch.long, device=device)
            idx_block_2d = idx_block.unsqueeze(0).expand(B, -1)
            shape_target = (B, position_end, -1)
            mask_key = mask_attention[:, :position_end]

            for step in range(step_per_block):

                if remainder_prompt and step != 0 and step % remainder_prompt == 0:
                    model(x[:, idx_prompt], idx_current=idx_prompt, shape_target=shape_target,
                          skip_logits=True, attention_mask=mask_key)

                if step == 0 or step % step_refresh_remainder == 0:
                    if step == 0 and idx_transform_2d is not None:
                        idx_union_prev = torch.unique(idx_transform_2d.flatten())
                        idx_current = torch.cat([idx_union_prev, idx_block])
                        mask_rows = torch.zeros(B, position_end, dtype=torch.bool, device=device)
                        mask_rows[:, idx_block] = True
                        mask_rows.scatter_(1, idx_transform_2d, True)
                        CachePastKVPlugin_Enabled.set_row_mask(mask_rows)
                    else:
                        idx_current = idx_block

                    logits = model(x[:, idx_current], idx_current=idx_current,
                                   shape_target=shape_target,
                                   attention_mask=mask_key).logits
                    CachePastKVPlugin_Enabled.set_row_mask(None)
                    logits_denoising = logits[:, -size_block:]

                    x_accumulated = x[:, :position_end]

                    def pad_to(table):
                        pad = torch.zeros((table.shape[0], position_end - table.shape[1]),
                                          dtype=table.dtype, device=table.device)
                        return torch.cat([table, pad], dim=1)

                    snapshot = SimpleLogitsSnapshot(
                        x_accumulated, x_accumulated, id_mask,
                        pad_to(snapshot.x0), pad_to(snapshot.conf),
                        pad_to(snapshot.margin), pad_to(snapshot.age))
                    snapshot.update_x0_(idx_block_2d, logits_denoising)
                    conf_snapshot = snapshot.transform_logits(
                        collector, logits_denoising, idx_transform=idx_block_2d)
                else:
                    score_attn_layers = plugin_cache_attn.collect_attn_from_all_blocks_batched(model)
                    idx_in_attn_2d = idx_transform_2d - position_start
                    mask_still = x[:, position_start:position_end] == id_mask

                    rows_denoising = []
                    for b in range(B):
                        attn_rows_all = score_attn_layers[:, b, idx_in_attn_2d[b], -size_block:]\
                            .mean(dim=1)
                        idx_local = select_topk_candidates(
                            router, spec_router,
                            attn_rows_all.float(),
                            snapshot.conf[b, position_start:position_end].float(),
                            mask_still[b],
                            idx_in_attn_2d[b, -1], h,
                            margin_block=snapshot.margin[b, position_start:position_end].float(),
                            age_block=snapshot.age[b, position_start:position_end].float(),
                        )
                        rows_denoising.append(idx_local + position_start)
                    idx_denoising_2d = torch.stack(rows_denoising, dim=0)

                    idx_union = torch.unique(torch.cat(
                        [idx_transform_2d.flatten(), idx_denoising_2d.flatten()]))
                    mask_rows = torch.zeros(B, position_end, dtype=torch.bool, device=device)
                    mask_rows.scatter_(1, idx_transform_2d, True)
                    mask_rows.scatter_(1, idx_denoising_2d, True)
                    CachePastKVPlugin_Enabled.set_row_mask(mask_rows)
                    logits_union = model(x[:, idx_union], idx_current=idx_union,
                                         shape_target=shape_target,
                                         attention_mask=mask_key).logits
                    CachePastKVPlugin_Enabled.set_row_mask(None)

                    lut = torch.full((position_end,), -1, dtype=torch.long, device=device)
                    lut[idx_union] = torch.arange(idx_union.shape[0], device=device)
                    rows = lut[idx_denoising_2d]
                    logits_transform = logits_union.gather(
                        1, rows.unsqueeze(-1).expand(-1, -1, logits_union.shape[-1]))

                    snapshot.update_x0_(idx_denoising_2d, logits_transform)
                    conf_snapshot = snapshot.transform_logits(
                        collector, logits_transform, idx_transform=idx_denoising_2d)

                    if future_idx_selector.select_only_in_h:
                        mask_denoising_no = torch.ones(
                            (B, conf_snapshot.shape[-1]), dtype=torch.bool, device=device)
                        mask_denoising_no.scatter_(1, idx_denoising_2d, False)
                        conf_snapshot.masked_fill_(mask_denoising_no,
                                                   torch.finfo(conf_snapshot.dtype).min)

                idx_sorted_by_conf = sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform_2d = idx_sorted_by_conf[:, :num_unmask]

                snapshot.materialize_by_idx_(idx_transform_2d, conf_snapshot)
                snapshot.update_this(1, idx_src=idx_transform_2d, x0=x)
                snapshot.tick_age_()

            for b in range(B):
                sentence_block = tokenizer.decode(x[b, idx_block])
                for word_stop in words_stop:
                    if word_stop in sentence_block:
                        has_done[b] = True

        for b in range(B):
            sentence_all = tokenizer.decode(x[b, len_prompt:position_end], skip_special_tokens=False)
            sentence_all = tokenizer.decode(tokenizer(sentence_all)['input_ids'],
                                            skip_special_tokens=True)
            for word_stop in words_stop:
                if word_stop in sentence_all:
                    sentence_all = sentence_all.split(word_stop)[0]
            sentences[b] = sentence_all

        return sentences, has_done

    def run_one(self, model, tokenizer, config, *args, **kwargs):
        import time

        config.klass_cache_attn.set_size_block(config.size_block)
        config.klass_cache_attn.set_len_prompt(kwargs['len_prompt'])

        path_router = getattr(config, 'path_router', None)
        assert path_router, 'the batch runner needs path_router'
        if self.router_bundle is None:
            from router_deploy import load_router_bundle
            from tools_debug import jprint
            router, spec_router = load_router_bundle(path_router, device=config.device)
            self.router_bundle = (router, spec_router)
            jprint(f'loaded router bundle {path_router}: {spec_router["features"]} '
                   f'norm={spec_router["normalization"]}')

        from future_idx_selector import FutureIDXSelector
        kwargs_selector = {}
        if config.h:
            kwargs_selector['h'] = config.h
        if config.select_only_in_h:
            kwargs_selector['select_only_in_h'] = config.select_only_in_h
        future_idx_selector = FutureIDXSelector(None, **kwargs_selector)

        plugin_cache_past_kv = config.klass_cache_past_kv()
        plugin_cache_attn = config.klass_cache_attn()
        plugin_cache_past_kv.clear(model)
        plugin_cache_attn.clear(model)

        kwargs['future_idx_selector'] = future_idx_selector
        kwargs['plugin_cache_attn'] = plugin_cache_attn
        kwargs['router_bundle'] = self.router_bundle

        time_start = time.perf_counter()
        sentences, has_done = self.generate(model, tokenizer, config, *args, **kwargs)
        duration_s = time.perf_counter() - time_start

        len_prompts = kwargs.get('len_prompts', [kwargs['len_prompt']] * len(sentences))
        for len_real, done_each in zip(len_prompts, has_done):
            self.report.add_and_dump(config, len_real, done_each,
                                     duration_s / len(sentences))

        return sentences, has_done
