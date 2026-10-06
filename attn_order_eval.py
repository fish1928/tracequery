import torch

class AttnOrderEval:

    def _build_geometry(self):
        T, L = self.T, self.L
        dev = self.attn.device

        step_of = torch.full((L,), -1, dtype=torch.long, device=dev)
        step_of[self.order] = torch.arange(T, device=dev)

        t_idx = torch.arange(T, device=dev).view(T, 1)
        future_gap = step_of.view(1, L) - t_idx
        cand_mask = future_gap > 0

        neg_inf = torch.finfo(self.attn.dtype).min
        S = self.attn.masked_fill(~cand_mask, neg_inf)

        n_cand = (T - 1) - torch.arange(T, device=dev)
        return future_gap, cand_mask, S, n_cand

    def __init__(self, attn_rows, order):
        assert attn_rows.dim() == 2, "attn_rows.dim(): {} == 2 false".format(attn_rows.dim())
        assert order.dim() == 1, "order.dim(): {} == 1 false".format(order.dim())

        self.attn = attn_rows.to(torch.float64)
        self.order = order.to(torch.long)

        self.T, self.L = self.attn.shape

        self.future_gap, self.cand_mask, self.S, self.n_cand = self._build_geometry()

    def get_attn(self):
        return self.attn

    def get_order(self):
        return self.order

    def get_future_gap(self):
        return self.future_gap

    def get_S(self):
        return self.S

    def get_n_cand(self):
        return self.n_cand

    def _nan_invalid(self, value, valid):
        nan = torch.full_like(value, float("nan"))
        return torch.where(valid, value, nan)

    def recall_at_h(self, h):
        rel = (self.future_gap >= 1) & (self.future_gap <= h)
        toph = self.S.topk(h, dim=-1).indices
        hit = rel.gather(-1, toph).double().sum(-1)
        recall = hit / h

        valid = self.n_cand >= h
        return self._nan_invalid(recall, valid)

    def pr_auc(self, h):
        idx_sorted = self.S.argsort(dim=-1, descending=True)
        gap_sorted = self.future_gap.gather(-1, idx_sorted)

        is_cand = gap_sorted > 0
        is_pos = (gap_sorted >= 1) & (gap_sorted <= h)

        tp = is_pos.cumsum(-1).double()
        retrieved = is_cand.cumsum(-1).double().clamp_min(1.0)
        precision = tp / retrieved

        P = is_pos.double().sum(-1)
        ap = (precision * is_pos.double()).sum(-1) / P.clamp_min(1.0)

        valid = self.n_cand > h
        return self._nan_invalid(ap, valid)

    def ndcg_at_h(self, H, gain="linear"):
        dev = self.attn.device

        base = (H - (self.future_gap - 1)).clamp(min=0).double()
        if gain == "exp":
            base = torch.exp2(base) - 1.0
        grade = base * self.cand_mask.double()

        disc = 1.0 / torch.log2(torch.arange(H, device=dev).double() + 2.0)
        pred = self.S.topk(H, dim=-1).indices
        dcg = (grade.gather(-1, pred) * disc).sum(-1)

        ideal = grade.topk(H, dim=-1).values
        idcg = (ideal * disc).sum(-1)

        ndcg = dcg / idcg.clamp_min(torch.finfo(torch.float64).tiny)
        valid = (self.n_cand >= 2) & (idcg > 0)
        return self._nan_invalid(ndcg, valid)

class MarginOrderEval(AttnOrderEval):

    def __init__(self, margin, order):
        assert margin.dim() == 2, "margin.dim(): {} == 2 false".format(margin.dim())

        super().__init__(margin, order)

    def get_margin(self):
        return self.attn


class ConfidenceOrderEval(AttnOrderEval):

    def __init__(self, confidence, order):
        assert confidence.dim() == 2, "confidence.dim(): {} == 2 false".format(confidence.dim())

        super().__init__(confidence, order)

    def get_confidence(self):
        return self.attn


class ScoreOrderEval(AttnOrderEval):

    def __init__(self, score, order):
        super().__init__(score, order)

    def get_score(self):
        return self.attn


def summ(x):
    return "{:.3f} (n={})".format(x.nanmean().item(), int((~x.isnan()).sum()))
