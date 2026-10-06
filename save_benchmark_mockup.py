import csv
import json
import os

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from tools_debug import jprint


FIELDNAMES_MOCKUP = ['id_request', 'task_name', 'doc_id', 'prompt', 'until', 'doc']


@register_model("mockup")
class MockupCollectorLM(LM):

    def __init__(self, batch_size=1, percent=0.1, folder_output='benchmark_mockup', tag='', merge='', *args, **kwargs):
        super().__init__()

        self.percent = float(percent)
        self.folder_output = folder_output
        self.tag = tag
        self.merge = merge

        assert 0.0 < self.percent <= 1.0, f'percent must be in (0, 1], got {self.percent}'

    def _build_path_output(self, name):
        name_parts = ['mockup', name]
        if self.tag:
            name_parts.append(self.tag)
        name_parts.append(f'p{int(self.percent * 100)}')

        return os.path.join(self.folder_output, '_'.join(name_parts) + '.csv')

    def _collect_task(self, task_name, requests_task):
        ids_doc = sorted({request_eval.doc_id for request_eval in requests_task})
        n_total = len(ids_doc)
        n_keep = max(1, int(n_total * self.percent))
        ids_keep = set(ids_doc[n_total - n_keep:])

        rows = []
        for id_request, request_eval in enumerate(requests_task):
            if request_eval.doc_id not in ids_keep:
                continue

            kwargs_gen = request_eval.args[1] if len(request_eval.args) > 1 else {}
            rows.append({
                'id_request': id_request,
                'task_name': task_name,
                'doc_id': request_eval.doc_id,
                'prompt': request_eval.args[0],
                'until': json.dumps(kwargs_gen.get('until', [])),
                'doc': json.dumps(getattr(request_eval, 'doc', {}), default=str),
            })

        meta = {
            'task_name': task_name,
            'mode': 'tail',
            'percent': self.percent,
            'n_total_docs': n_total,
            'n_keep_docs': n_keep,
            'n_rows': len(rows),
            'limit_safe_max': n_total - n_keep,
            'ids_doc_keep': sorted(ids_keep),
        }
        return rows, meta

    def _write_csv(self, path_output, rows, metas):
        os.makedirs(os.path.dirname(path_output) or '.', exist_ok=True)

        with open(path_output, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES_MOCKUP, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

        with open(path_output + '.meta.json', 'w') as file:
            json.dump({'tasks': metas}, file, indent=2)

        n_docs = sum(meta['n_keep_docs'] for meta in metas)
        n_total = sum(meta['n_total_docs'] for meta in metas)
        jprint(f'saved {n_docs}/{n_total} docs ({len(rows)} rows, {len(metas)} task(s)) to {path_output}')

    def generate_until(self, requests_eval):
        tasks = {}
        for request_eval in requests_eval:
            name = getattr(request_eval, 'task_name', 'unknown')
            tasks.setdefault(name, []).append(request_eval)

        if self.merge:
            rows_all, metas_all = [], []
            for task_name in sorted(tasks):
                rows, meta = self._collect_task(task_name, tasks[task_name])
                rows_all.extend(rows)
                metas_all.append(meta)
            self._write_csv(self._build_path_output(self.merge), rows_all, metas_all)
        else:
            for task_name in sorted(tasks):
                rows, meta = self._collect_task(task_name, tasks[task_name])
                self._write_csv(self._build_path_output(task_name), rows, [meta])

        return [''] * len(requests_eval)

    def loglikelihood(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError('mockup collector only supports generate_until tasks')


def load_benchmark_mockup(path_csv, filter_task=None):
    rows = []
    with open(path_csv, 'r', newline='') as file:
        for row in csv.DictReader(file):
            if filter_task is not None and row['task_name'] != filter_task:
                continue
            row['until'] = json.loads(row['until'])
            row['doc'] = json.loads(row['doc'])
            row['id_request'] = int(row['id_request'])
            row['doc_id'] = int(row['doc_id'])
            rows.append(row)
    return rows


if __name__ == "__main__":
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
    cli_evaluate()
