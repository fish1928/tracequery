import torch
import torch.nn.functional as F
import os

from tools_debug import jprint

class SimpleLogitsSnapshot:

    TRACE = None

    def __init__(self, x, y, id_mask, x0=None, conf=None, margin=None, age=None):
        self.id_mask = id_mask

        self.x = x
        self.y = y

        assert x0 is None or x0.shape[1] == self.x.shape[1]
        assert conf is None or conf.shape[1] == self.x.shape[1]
        assert margin is None or margin.shape[1] == self.x.shape[1]
        assert age is None or age.shape[1] == self.x.shape[1]

        if x0 is None:
            self.x0 = torch.zeros(self.x.shape, dtype=torch.long, device=self.x.device)
        else:
            self.x0 = x0

        self.p_finalized = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)

        if conf is None:
            self.conf = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)
        else:
            self.conf = conf

        if margin is None:
            self.margin = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)
        else:
            self.margin = margin

        if age is None:
            self.age = torch.zeros(self.x.shape, dtype=torch.float32, device=self.x.device)
        else:
            self.age = age

    def tick_age_(self):
        self.age += 1.0
        return self

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_p_finalized(self):
        return self.p_finalized


    def materialize_by_idx_(self, idx, conf):

        x0_target = torch.gather(self.x0, dim=-1, index=idx)
        conf_target = torch.gather(conf, dim=-1, index=idx)
        self.x.scatter_(1, idx, x0_target)
        self.p_finalized.scatter_(1, idx, conf_target)

    def update_x0_(self, idx_transform, logits):
        assert idx_transform.dim() == 2, "idx_transform.dim(): {} == 2 false".format(idx_transform.dim())

        x0 = torch.argmax(logits, dim=-1)
        self.x0.scatter_(1, idx_transform, x0)
        return idx_transform


    def transform_logits(self, collector, logits, idx_transform=None):
        index_p_transform = collector.get_index(self, idx_transform)
        p_transformed = F.softmax(logits.float(), dim=-1)

        x0_p_transformed = torch.gather(p_transformed, dim=-1, index=index_p_transform).squeeze(-1)

        top2 = p_transformed.topk(2, dim=-1).values
        margin_transformed = top2[..., 0] - top2[..., 1]

        if idx_transform is not None:
            self.conf.scatter_(1, idx_transform, x0_p_transformed)
            self.margin.scatter_(1, idx_transform, margin_transformed)
            self.age.scatter_(1, idx_transform, torch.zeros_like(x0_p_transformed))
        else:
            self.conf = x0_p_transformed
            self.margin = margin_transformed
            self.age = torch.zeros_like(self.conf)

        neg_inf = torch.tensor(
            torch.finfo(self.conf.dtype).min,
            device=self.conf.device,
            dtype=self.conf.dtype
        )

        mask_mask = self.x == self.id_mask

        return torch.where(mask_mask, self.conf, neg_inf)

    def update_this(self, dim, idx_src, idx_tgt=None, **kwargs):

        if idx_tgt is None:
            idx_transform = idx_src
        else:
            idx_tgt=idx_tgt.unsqueeze(0)
            
            idx_transform = torch.gather(idx_tgt, dim=-1, index=idx_src)

        for k, v in kwargs.items():
            v.scatter_(dim, idx_transform, torch.gather(getattr(self, k), dim=dim, index=idx_src))

        if SimpleLogitsSnapshot.TRACE is not None and 'x0' in kwargs:
            SimpleLogitsSnapshot.TRACE.append((
                idx_transform[0].detach().cpu().numpy().copy(),
                torch.gather(self.x0, dim, idx_src)[0].detach().cpu().numpy().copy()))

        return self


class IndexedElementList:
    def __init__(self, idx_start, idx_end, name=None):
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.indexed_elements = [None] * (idx_end - idx_start)
        self.name = name

    def add(self, idx_relative, element):
        self.indexed_elements[idx_relative - self.idx_start] = element

    def get(self, idx_relative):
        return self.indexed_elements[idx_relative - self.idx_start]

    def has_empty(self):
        for indexed_element in self.indexed_elements:
            if indexed_element is None:
                return True

        return False

    def stack_and_save(self, path_to_save):
        stacked_elements = torch.stack(self.indexed_elements)
        torch.save(stacked_elements, os.path.join(path_to_save, f'{self.name}_{self.idx_start}_{self.idx_end}.pt'))


class Stats:
    _STATS = ('margin', 'conf', 'attn', 'unmask')

    def __init__(self, idx_start, idx_end, names=None, margin_idx_a=0, margin_idx_b=1):
        self.names = tuple(names) if names is not None else self._STATS

        for name in self.names:
            setattr(self, name, IndexedElementList(idx_start, idx_end, name=name))

        if 'margin' in self.names:
            self.margin.idx_a = margin_idx_a
            self.margin.idx_b = margin_idx_b

    def stack_and_save_all(self, path_to_save):
        for name_stat in self.names:
            statlist = getattr(self, name_stat)
            if not statlist.has_empty():
                statlist.stack_and_save(path_to_save)
