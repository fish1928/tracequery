import torch
from abc import ABC, abstractmethod


class Preprocessor_(ABC):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def _tokenize(self, ds_each):
        pass

    def __call__(self, ds_each):
        return self._tokenize(ds_each)

class Preprocessor_Until(Preprocessor_):

    def __init__(self, tokenizer, use_chat_template=False, use_official_gsm8k_prompt=False):
        super().__init__(tokenizer)
        self.use_chat_template = use_chat_template
        self.use_official_gsm8k_prompt = use_official_gsm8k_prompt

    def _tokenize(self, ds_each):
        text_prompt = ds_each['prompt']

        if self.use_official_gsm8k_prompt:
            from run_official_llada_gsm8k import build_messages

            question = text_prompt
            if question.startswith('Question: '):
                question = question[len('Question: '):]
            if question.rstrip().endswith('Answer:'):
                question = question.rstrip()[:-len('Answer:')].rstrip()

            text_prompt = self.tokenizer.apply_chat_template(
                build_messages(question, num_fewshot=4),
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = self.tokenizer(text_prompt)['input_ids']

            return {
                'ids_prompt': ids,
                'text_prompt': text_prompt,
                'until': ds_each['until']
            }

        if self.use_chat_template:
            text_prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text_prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        ids = self.tokenizer(
            text_prompt,
            add_special_tokens=False
        )["input_ids"]

        return {
            'ids_prompt': ids,
            'text_prompt': text_prompt,
            'until': ds_each['until']
        }


class Collater_(ABC):
    @abstractmethod
    def _collate(self, ds_batch):
        pass

    def __call__(self, ds_batch):
        return self._collate(ds_batch)

class Collater_Until_One(Collater_):

    def __init__(self, config):
        self.len_target = config.len_target
        self.id_mask = config.id_mask

    def _collate(self, ds_batch):
        if type(ds_batch) is list:
            ds_batch = ds_batch[0]

        ids_prompt = ds_batch['ids_prompt']
        len_prompt = len(ids_prompt)

        ids_input = ids_prompt + [self.id_mask] * self.len_target
        ids_input = torch.tensor(ids_input, dtype=torch.long).view(1, -1)

        return {
            'ids_input': ids_input,
            'text_prompt': ds_batch['text_prompt'],
            'len_prompt': len_prompt,
            'until': ds_batch['until']
        }


class Collater_Until_Batch(Collater_):

    def __init__(self, config, id_pad):
        self.len_target = config.len_target
        self.id_mask = config.id_mask
        assert id_pad is None or id_pad != config.id_mask, 'pad id must differ from the mask id'
        self.id_pad = id_pad

    def _collate(self, ds_batch):
        assert type(ds_batch) is list and len(ds_batch) >= 1
        len_prompts = [len(ds_each['ids_prompt']) for ds_each in ds_batch]
        len_prompt_pad = max(len_prompts)
        if len_prompt_pad != min(len_prompts):
            assert self.id_pad is not None, 'batched collation needs a pad token id'

        rows = []
        masks = []
        for ds_each, len_real in zip(ds_batch, len_prompts):
            num_pad = len_prompt_pad - len_real
            ids_row = [self.id_pad] * num_pad + ds_each['ids_prompt'] \
                + [self.id_mask] * self.len_target
            rows.append(ids_row)
            masks.append([0] * num_pad + [1] * (len_real + self.len_target))

        until = ds_batch[0]['until']
        assert all(ds_each['until'] == until for ds_each in ds_batch), \
            'until must be identical within a batch (same task guarantees this)'

        return {
            'ids_input': torch.tensor(rows, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
            'len_prompt': len_prompt_pad,
            'len_prompts': len_prompts,
            'text_prompt': [ds_each['text_prompt'] for ds_each in ds_batch],
            'until': until,
        }


class Collater_sample(Collater_):

    def __init__(self, id_mask):
        self.id_mask = id_mask

    def _collate(self, ds_batch):
        if type(ds_batch) is list:
            ds_batch = ds_batch[0]

        x = torch.tensor(ds_batch['x'], dtype=torch.long)
        y = torch.tensor(ds_batch['y'], dtype=torch.long)

        assert x.shape == y.shape

        len_prompt = (x != self.id_mask).sum().item()

        return {
            'ids_prompt_masked_full': x,
            'ids_target_masked_full': y,
            'len_prompt': len_prompt
        }
