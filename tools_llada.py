import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tools_debug import jprint


class DiffusionQuotaHelper(ABC):
    @abstractmethod
    def get_quota(self, step_current):
        pass

class BlockDiffusionQuotaHelper(DiffusionQuotaHelper):
    def __init__(self, block_mask_index: torch.Tensor, steps_per_block: int) -> torch.Tensor:
        device = block_mask_index.device
        dtype = torch.long

        total = block_mask_index.sum(dim=1)
        base  = torch.div(total, steps_per_block, rounding_mode='floor')
        rem   = total - base * steps_per_block

        num_transfer_tokens = base.unsqueeze(1).expand(-1, steps_per_block).to(dtype)

        cols = torch.arange(steps_per_block, device=device).unsqueeze(0)
        add_mask = cols < rem.unsqueeze(1)

        self.num_transfer_tokens = (num_transfer_tokens + add_mask.to(dtype)).cpu()

    def get_quota(self, step_current):
        quota_current = self.num_transfer_tokens[:, step_current]

        if quota_current.dim() == 2 and quota_current.size(1) == 1:
            quota_current = quota_current.squeeze(1)

        return int(quota_current.max())

    def get_quota_max(self):
        return int(self.num_transfer_tokens.max())


class ConfKSorter:

    def argsort(self, conf_all):
        idx_sorted = torch.argsort(conf_all, dim=1, descending=True)
        return idx_sorted

class RandomKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):

        confidence = torch.where(
                snapshot.mask_mask,
                torch.rand(confidence.shape[0], confidence.shape[1], device=confidence.device),
                confidence
            )

        return super().argsort(confidence)


class TopKSorter(ConfKSorter):
    def argsort(self, confidence, snapshot):
        return super().argsort(confidence)


class ConfCollectorInterface(ABC):
    @abstractmethod
    def get_index(self, snapshot, idx=None):
        pass

class TruthCollector(ConfCollectorInterface):
    def get_index(self, snapshot, idx=None):
        index = snapshot.y

        if idx is not None:
            index = torch.gather(index, 1, idx)

        return index.unsqueeze(-1)


class MaxCollector(ConfCollectorInterface):
    def get_index(self, snapshot, idx=None):
        index = snapshot.x0

        if idx is not None:
            index = torch.gather(index, 1, idx)
        return index.unsqueeze(-1)


class LogitsTransformer:
    def transform_logits(self, logits, collector):
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = collector.gather_x0_p(p, self)
        return x0_p


class PPLCalculator:
    def cal(self, probs_all, mask_target=None, eps=1e-12):
        if mask_target is None:
            mask_target = slice(None)

        probs_collected = probs_all[mask_target].reshape(-1)

        mean_prob = probs_collected.mean(dim=-1)

        nll_collected = -torch.log(probs_collected + eps)
        nll_per = nll_collected.mean(dim=-1)
        ppl_per = torch.exp(nll_per)

        return ppl_per.item(), mean_prob.item()


class RefreshIdxHelper:
    TYPE_HIDDEN = {
        'k':'_k_previous',
        'v':'_v_previous'
    }

    def __init__(self, dict_filename_to_list_idx_sorted, type_hidden_str, size_block, randomed=False):
        self.dict_filename_to_list_idx_sorted = dict_filename_to_list_idx_sorted
        self.type_hidden=RefreshIdxHelper.TYPE_HIDDEN[type_hidden_str]
        self.size_block = size_block
        self.randomed = randomed

    def set_budget(self, budget):
        self.budget = budget
        return self

    def set_sample_id(self, id_sample):
        self.id_sample = id_sample
        return self

    def set_randomed(self, randomed):
        self.randomed = randomed

    def get_refresh_idx(self, x, id_step, id_block, return_sorted=True, id_step_global=None):
        id_sample = self.id_sample
        budget = self.budget
        size_block = self.size_block

        if id_step_global is None:
            id_step_global = id_step + id_block * size_block

        randomed = self.randomed

        filename = f'batch_{id_sample}{self.type_hidden}.pt'
        list_step_list_idx_sorted = self.dict_filename_to_list_idx_sorted[filename]

        assert list_step_list_idx_sorted[id_step_global]['step'] == id_step,\
            f'{list_step_list_idx_sorted[id_step_global]["step"]} == {id_step}'

        list_idx_sorted = list_step_list_idx_sorted[id_step_global]['idx']

        if budget < 1.0:
            budget = int(len(list_idx_sorted) * budget) or 1

        list_idx_sorted = torch.tensor(list_idx_sorted, dtype=torch.long, device=x.device)

        if randomed:
            idxs_list_idx_rand = torch.randperm(list_idx_sorted.shape[0])
            list_idx_sorted = list_idx_sorted[idxs_list_idx_rand]

        result = list_idx_sorted[:budget]

        return torch.sort(result)[0] if return_sorted else result


def collect_ids_stop(tokenizer):
    ids_stop = set()
    if tokenizer.eos_token_id is not None:
        ids_stop.add(tokenizer.eos_token_id)

    for token_stop in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
        id_token = tokenizer.convert_tokens_to_ids(token_stop)
        if id_token is not None and id_token >= 0 and id_token != tokenizer.unk_token_id:
            ids_stop.add(id_token)

    return ids_stop


def truncate_text_at_stop(tokenizer, ids_generated, ids_stop, words_stop):
    has_done = False

    tensor_stop = torch.tensor(sorted(ids_stop), dtype=ids_generated.dtype, device=ids_generated.device)
    hits_stop = torch.isin(ids_generated, tensor_stop).nonzero()
    if hits_stop.numel() > 0:
        ids_generated = ids_generated[:hits_stop[0, 0]]
        has_done = True

    text = tokenizer.decode(ids_generated, skip_special_tokens=True)
    for word_stop in words_stop:
        if word_stop in text:
            text = text.split(word_stop)[0]
            has_done = True

    return text, has_done


def certainty_density(mask, sigma):
    assert sigma > 0
    B, L = mask.shape
    device = mask.device
    float_mask = mask.float()

    padded_mask = F.pad(float_mask, (L, L), 'constant', 1.0)
    padded_mask[mask[:, -1] == False, 2 * L:] = 0.0

    extended_L = 3 * L
    padded_len = 2 * extended_L

    dist = torch.cat((
        torch.arange(extended_L, device=device),
        torch.arange(-extended_L, 0, device=device),
    ))
    kernel_fft = torch.fft.fft(torch.exp(-(dist ** 2) / (2 * sigma ** 2)), n=padded_len)

    weighted_sum_ext = torch.fft.ifft(
        torch.fft.fft(F.pad(padded_mask, (0, extended_L)), n=padded_len) * kernel_fft,
        n=padded_len).real
    kernel_sum_ext = torch.fft.ifft(
        torch.fft.fft(torch.ones(B, extended_L * 2, device=device), n=padded_len) * kernel_fft,
        n=padded_len).real

    weighted_sum = weighted_sum_ext[..., L:2 * L]
    kernel_sum_at_pos = kernel_sum_ext[..., L:2 * L].clamp_min(1e-8)
    return weighted_sum / kernel_sum_at_pos


def nucleus_select(scores, top_p, min_k=1, mask=None):
    scores = torch.where(mask, scores, torch.zeros_like(scores)) if mask is not None else scores

    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= top_p

    k = min(min_k, scores.shape[-1])
    top_k_mask = torch.arange(nucleus_mask.shape[-1], device=nucleus_mask.device) < k

    combined_mask = nucleus_mask | top_k_mask
    if mask is not None:
        combined_mask &= torch.gather(mask, 1, sorted_indices)

    return torch.zeros_like(scores, dtype=torch.bool).scatter_(
        dim=1, index=sorted_indices, src=combined_mask)


def inflate_selection(q_mask, inflate_w):
    if inflate_w <= 0:
        return q_mask
    B, T = q_mask.shape
    device = q_mask.device
    arange_t = torch.arange(T, device=device).expand(B, -1)

    masked_next = torch.where(q_mask, arange_t, T)
    next_sel = torch.flip(torch.cummin(torch.flip(masked_next, dims=[-1]), dim=-1).values, dims=[-1])
    dist_next = next_sel - arange_t

    masked_prev = torch.where(q_mask, arange_t, -1)
    prev_sel = torch.cummax(masked_prev, dim=-1).values
    dist_prev = arange_t - prev_sel

    gap_len = dist_next + dist_prev
    return q_mask | ((gap_len <= inflate_w) & (prev_sel >= 0) & (next_sel < T))


class RunnerReport:
    def __init__(self):
        self.rows = []

    def add_and_dump(self, config, len_prompt, has_done, duration_s):
        path_report = getattr(config, 'path_report', None)
        if not path_report:
            return

        mem_alloc_gib = mem_reserved_gib = None
        try:
            import torch
            if torch.cuda.is_available():
                mem_alloc_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                mem_reserved_gib = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        except Exception:
            pass

        self.rows.append({
            'id_sample': len(self.rows),
            'len_prompt': len_prompt,
            'has_done': has_done,
            'duration_s': round(duration_s, 4),
            'mem_alloc_gib': mem_alloc_gib,
            'mem_reserved_gib': mem_reserved_gib,
        })

        import json
        with open(path_report, 'w') as file:
            json.dump({
                'path_router': getattr(config, 'path_router', None),
                'num_samples': len(self.rows),
                'duration_total_s': round(sum(row['duration_s'] for row in self.rows), 2),
                'rows': self.rows,
            }, file, indent=2)
