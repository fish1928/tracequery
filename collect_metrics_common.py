import argparse
import json
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from components_llada import Stats
from tools_llada import TopKSorter, MaxCollector, collect_ids_stop, truncate_text_at_stop
from plugins_llada import SaveKVPreviousPlugin_Disabled,\
                            CachePastKVPlugin_Disabled,\
                            CacheAttnPlugin_Enabled,\
                            CacheVOPlugin_Disabled
from dataprocess_llada import Preprocessor_Until
from save_benchmark_mockup import load_benchmark_mockup
from constants_llada import DTYPE_EVAL
from tools_debug import jprint


NAMES_STATS = ('margin', 'conf', 'entropy', 'attn', 'unmask', 'token', 'x0')


def build_parser(id_model, id_mask, len_target=256, num_blocks=1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_mockup', type=str, required=True)
    parser.add_argument('--folder_output', type=str, required=True)
    parser.add_argument('--id_model', type=str, default=id_model)
    parser.add_argument('--len_target', type=int, default=len_target)
    parser.add_argument('--num_blocks', type=int, default=num_blocks)
    parser.add_argument('--id_mask', type=int, default=id_mask)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--limit', type=int, default=None, help='cap on mockup rows')
    parser.add_argument('--row_start', type=int, default=0,
                        help='skip the first N mockup rows before --limit; sample folders '
                             'are still numbered from 0 (row_start=450 --limit 50 recreates '
                             'make_train_split\'s head-split training tail directly)')
    parser.add_argument('--tail_percent', type=float, default=None,
                        help='keep only the LAST fraction of each task_name group before '
                             'applying --limit. REQUIRED when the mockup was collected with '
                             'percent=1 (full benchmark): recreates the held-out-tail design '
                             'so oracle/training docs stay disjoint from eval docs (lm_eval '
                             '--limit evaluates the FIRST docs). Use 0.1 for p100 mockups; '
                             'leave unset for tail mockups (p10), which are already tails.')
    parser.add_argument('--filter_task', type=str, default=None,
                        help='keep only this subtask from a merged mockup CSV (e.g. one bbh subtask)')
    parser.add_argument('--plain_prompt', action='store_true',
                        help='force use_chat_template=False even on instruct collectors: '
                             'code tasks (humaneval/mbpp) are completion tasks -- lm_eval '
                             'executes prompt+generation as one unit, so chat-wrapped '
                             'answers are structurally unscorable')
    parser.add_argument('--seed', type=int, default=233)
    return parser


def check_result_gsm8k(text_checked, doc):
    match_gold = re.search(r'####\s*(-?[0-9\.,]+)', str(doc.get('answer', '')))
    if match_gold is None:
        return 'unknown'
    gold = match_gold.group(1).replace(',', '').rstrip('.')

    nums = re.findall(r'-?[0-9][0-9,]*\.?[0-9]*', text_checked.replace('$', ''))
    if not nums:
        return 'fail'
    pred = nums[-1].replace(',', '').rstrip('.')

    return 'pass' if pred == gold else 'fail'


def check_result_ifeval(text_checked, doc):
    try:
        from lm_eval.tasks.ifeval.utils import process_results
        scores = process_results(doc, [text_checked])
    except Exception as error:
        jprint(f'ifeval checker unavailable or failed: {error}')
        return 'unknown'

    result = 'pass' if scores.get('prompt_level_strict_acc') else 'fail'
    return result, scores


def check_result_minerva_math(text_checked, doc):
    try:
        from lm_eval.tasks.minerva_math.utils import process_results
        scores = process_results(doc, [text_checked])
    except Exception as error:
        jprint(f'minerva_math checker unavailable or failed: {error}')
        return 'unknown'

    return ('pass' if scores.get('exact_match') else 'fail'), scores


def check_result_bbh(text_checked, doc):
    target = doc.get('target')
    if target is None:
        return 'unknown'

    match_pred = re.search(r'the answer is\s*(.+?)(?:\.|$)', text_checked, re.IGNORECASE)
    if match_pred is None:
        return 'fail'

    def normalize(text):
        return text.strip().strip('.').strip('()').strip().casefold()

    return 'pass' if normalize(match_pred.group(1)) == normalize(str(target)) else 'fail'


MAP_TASK_CHECKER = {
    'gsm8k': check_result_gsm8k,
    'ifeval': check_result_ifeval,
}

PREFIXES_TASK_CHECKER = (
    ('minerva_math', check_result_minerva_math),
    ('bbh', check_result_bbh),
)


def resolve_checker(task_name):
    if task_name in MAP_TASK_CHECKER:
        return MAP_TASK_CHECKER[task_name]
    for prefix, checker in PREFIXES_TASK_CHECKER:
        if task_name.startswith(prefix):
            return checker
    return None


def summarize_records(records):
    summary = {}

    def _bucket(records_bucket):
        n_pass = sum(1 for r in records_bucket if r['result'] == 'pass')
        n_fail = sum(1 for r in records_bucket if r['result'] == 'fail')
        n_unknown = sum(1 for r in records_bucket if r['result'] == 'unknown')
        n_scored = n_pass + n_fail
        entry = {
            'n': len(records_bucket),
            'pass': n_pass,
            'fail': n_fail,
            'unknown': n_unknown,
            'accuracy': round(n_pass / n_scored, 4) if n_scored else None,
            'has_done_rate': round(sum(1 for r in records_bucket if r['has_done']) / len(records_bucket), 4),
        }

        details = [r['result_detail'] for r in records_bucket if r.get('result_detail')]
        if details:
            keys = sorted(set().union(*(d.keys() for d in details)))
            for key in keys:
                values = [float(d[key]) for d in details
                          if isinstance(d.get(key), (int, float, bool))]
                if values:
                    entry[f'mean_{key}'] = round(sum(values) / len(values), 4)
        return entry

    names_task = sorted({r['task_name'] for r in records})
    for name_task in names_task:
        summary[name_task] = _bucket([r for r in records if r['task_name'] == name_task])
    if len(names_task) > 1:
        summary['__overall__'] = _bucket(records)

    return summary


class OracleCollectorBase:

    use_chat_template = False

    def __init__(self, args, klass_model):
        self.args = args

        if getattr(args, 'plain_prompt', False):
            self.use_chat_template = False

        assert args.len_target % args.num_blocks == 0
        self.size_block = args.len_target // args.num_blocks
        self.step_per_block = self.size_block
        self.sorter = TopKSorter()
        self.collector = MaxCollector()

        self.tokenizer = AutoTokenizer.from_pretrained(args.id_model, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        assert self.tokenizer.pad_token_id != args.id_mask

        self.model = klass_model.from_pretrained(
            args.id_model,
            trust_remote_code=True,
            torch_dtype=DTYPE_EVAL,
        ).eval().to(args.device)

        self.model\
            .fill_plugin(CachePastKVPlugin_Disabled)\
            .fill_plugin(SaveKVPreviousPlugin_Disabled)\
            .fill_plugin(CacheAttnPlugin_Enabled)\
            .fill_plugin(CacheVOPlugin_Disabled)

        self.plugin_cache_attn = CacheAttnPlugin_Enabled()

        self.preprocessor = Preprocessor_Until(
            self.tokenizer,
            use_chat_template=self.use_chat_template,
            use_official_gsm8k_prompt=bool(getattr(args, 'use_official_gsm8k_prompt', False)),
        )

        self.ids_stop = collect_ids_stop(self.tokenizer)

    def collect_one(self, x, len_prompt, folder_stats):
        raise NotImplementedError('thread collector must implement collect_one()')

    def run(self, rows):
        args = self.args
        records_summary = []

        for id_row, row in enumerate(tqdm(rows)):
            folder_stats = os.path.join(args.folder_output, str(id_row))
            path_generated = os.path.join(folder_stats, 'generated.json')

            if os.path.exists(path_generated):
                with open(path_generated, 'r') as file:
                    record = json.load(file)
                records_summary.append({
                    'task_name': record['task_name'],
                    'result': record['result'],
                    'has_done': record['has_done'],
                    'result_detail': record.get('result_detail'),
                })
                continue

            processed = self.preprocessor({'prompt': row['prompt'], 'until': row['until']})
            ids_prompt = processed['ids_prompt']
            len_prompt = len(ids_prompt)

            x = torch.tensor(ids_prompt + [args.id_mask] * args.len_target, dtype=torch.long).view(1, -1)
            x = x.to(args.device)

            CacheAttnPlugin_Enabled.set_len_prompt(len_prompt).set_size_block(self.size_block)
            self.plugin_cache_attn.clear(self.model)

            position_end = self.collect_one(x, len_prompt, folder_stats)

            text_generated = self.tokenizer.batch_decode(x[:, len_prompt:position_end], skip_special_tokens=False)[0]

            text_checked, has_done = truncate_text_at_stop(
                self.tokenizer, x[0, len_prompt:position_end], self.ids_stop, row['until'])

            checker = resolve_checker(row['task_name'])
            result, result_detail = 'unknown', None
            if checker is not None:
                result = checker(text_checked, row['doc'])
                if isinstance(result, tuple):
                    result, result_detail = result

            record = {
                'id_request': row['id_request'],
                'doc_id': row['doc_id'],
                'task_name': row['task_name'],
                'has_done': has_done,
                'result': result,
                'text_generated': text_generated,
            }
            if result_detail is not None:
                record['result_detail'] = result_detail

            with open(os.path.join(folder_stats, 'generated.json'), 'w') as file:
                json.dump(record, file)

            records_summary.append({
                'task_name': record['task_name'],
                'result': record['result'],
                'has_done': record['has_done'],
                'result_detail': record.get('result_detail'),
            })

        summary = summarize_records(records_summary)
        with open(os.path.join(args.folder_output, 'eval_summary.json'), 'w') as file:
            json.dump(summary, file, indent=2)
        jprint('=== oracle collection eval summary ({} samples) ==='.format(len(records_summary)))
        jprint(json.dumps(summary, indent=2))


def main_collect(klass_collector, klass_model, parser):
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    rows = load_benchmark_mockup(args.path_mockup, filter_task=args.filter_task)

    if args.tail_percent is not None:
        assert 0.0 < args.tail_percent <= 1.0
        groups = {}
        for row in rows:
            groups.setdefault(row['task_name'], []).append(row)
        rows = []
        for name_task in sorted(groups):
            rows_group = groups[name_task]
            n_keep = max(1, int(len(rows_group) * args.tail_percent))
            rows.extend(rows_group[-n_keep:])
        jprint('tail_percent={}: kept {} rows across {} task groups'.format(
            args.tail_percent, len(rows), len(groups)))

    row_start = args.row_start
    if args.limit is not None and row_start + args.limit > len(rows):
        row_start = max(0, len(rows) - args.limit)
        jprint(f'row_start {args.row_start} + limit {args.limit} > {len(rows)} rows: '
               f'clamped to the last {args.limit} rows (row_start={row_start})')
    rows = rows[row_start:]
    if args.limit is not None:
        rows = rows[:args.limit]
    jprint(f'collecting oracle for {len(rows)} prompts from {args.path_mockup} '
           f'(len_target={args.len_target}, num_blocks={args.num_blocks})')

    collector = klass_collector(args, klass_model)
    collector.run(rows)
