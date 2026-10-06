import os

import torch
import torch.nn.functional as F

from components_llada import SimpleLogitsSnapshot, Stats
from tools_llada import BlockDiffusionQuotaHelper
from modeling_llada_yukai_06 import LLaDAModelLM
from collect_metrics_common import NAMES_STATS, OracleCollectorBase, build_parser, main_collect


class OracleCollector(OracleCollectorBase):

    use_chat_template = False

    @torch.no_grad()
    def collect_one(self, x, len_prompt, folder_stats):
        args = self.args
        id_mask = args.id_mask
        size_block = self.size_block

        neg_inf = torch.finfo(torch.float32).min
        position_start = 0

        for id_block in range(args.num_blocks):
            position_end = position_start + len_prompt + (id_block + 1) * size_block
            block_start = position_end - size_block
            mask_mask_blk = x[:, position_start:position_end] == id_mask

            idx_denoising = torch.arange(position_start, position_end, dtype=torch.long, device=x.device)
            idx_block = torch.arange(block_start, position_end, dtype=torch.long, device=x.device)
            idx_block_2d = idx_block.unsqueeze(0)
            quota_helper = BlockDiffusionQuotaHelper(mask_mask_blk, self.step_per_block)
            shape_target = (x.shape[0], position_end, -1)

            stats = Stats(block_start, position_end, names=NAMES_STATS)

            for step in range(self.step_per_block):
                x_denoising, y_denoising = x[:, idx_denoising], x[:, idx_denoising]
                logits = self.model(x_denoising, idx_current=idx_denoising, shape_target=shape_target).logits

                logits_blk = logits[:, idx_block].float()
                mask_blk = (x[:, block_start:position_end] == id_mask).squeeze(0)
                sentinel = torch.tensor(neg_inf, device=x.device)

                logp_blk = F.log_softmax(logits_blk, dim=-1)
                p_blk = logp_blk.exp()

                top2 = p_blk.topk(2, dim=-1).values.squeeze(0)
                margin_blk = torch.where(mask_blk, top2[:, 0] - top2[:, 1], sentinel)
                stats.margin.add(block_start + step, margin_blk.cpu())

                entropy_blk = -(p_blk * logp_blk).sum(dim=-1).squeeze(0)
                entropy_blk = torch.where(mask_blk, entropy_blk, sentinel)
                stats.entropy.add(block_start + step, entropy_blk.cpu())

                stats.x0.add(block_start + step, logits_blk.argmax(dim=-1).squeeze(0).cpu())

                snapshot = SimpleLogitsSnapshot(x_denoising, y_denoising, id_mask)
                snapshot.update_x0_(idx_block_2d, logits_blk)
                conf_snapshot = snapshot.transform_logits(self.collector, logits_blk, idx_transform=idx_block_2d)
                stats.conf.add(block_start + step, conf_snapshot.squeeze(0)[block_start:position_end].cpu())

                idx_sorted_by_conf = self.sorter.argsort(conf_snapshot, snapshot)
                num_unmask = quota_helper.get_quota(step)
                idx_transform = idx_sorted_by_conf[:, :num_unmask]
                snapshot.materialize_by_idx_(idx_transform, conf_snapshot)

                attn_all = self.plugin_cache_attn.collect_attn_from_all_blocks(self.model)
                idx_local = idx_transform.squeeze(0) - block_start
                assert bool((idx_local >= 0).all() and (idx_local < size_block).all()),\
                    f'unmask position outside current block: {idx_transform.tolist()}'
                stats.attn.add(block_start + step, attn_all[:, idx_local, :].cpu())

                stats.unmask.add(block_start + step, idx_transform.squeeze(0).cpu())

                token_written = torch.gather(snapshot.x0, 1, idx_transform).squeeze(0)
                stats.token.add(block_start + step, token_written.cpu())

                snapshot.update_this(1, idx_transform, x0=x)

            os.makedirs(folder_stats, exist_ok=True)
            stats.stack_and_save_all(folder_stats)

        with open(os.path.join(folder_stats, '.pos_root'), 'w+') as file:
            file.write(str(len_prompt))

        return position_end


if __name__ == '__main__':
    main_collect(
        OracleCollector,
        LLaDAModelLM,
        build_parser(id_model='GSAI-ML/LLaDA-8B-Base', id_mask=126336),
    )
