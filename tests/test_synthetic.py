"""随机权重与 Graph 的实质边界；不把 CPU 替身当作 GPU 验收。"""
import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hicache import benchmark, metrics
from hicache.prepare import ASSET_FILES, prepare_model
from hicache.runner import Job
from hicache.metadata import snapshot_code


class SyntheticConfiguration(unittest.TestCase):
    def test_frozen_full_entry_runs_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            code = Path(temp) / 'moved-code'
            snapshot_code(code)
            result = subprocess.run([sys.executable, str(code / 'run_all.py'),
                                     'qwen35-35b-matrix', '--groups', 'l1', '--dry-run'],
                                    cwd=temp, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['server']['load-format'], 'dummy')
            self.assertFalse((code / 'results').exists())

    def job(self, **changes):
        settings = dict(experiment='qwen35-35b-matrix', runtime='native', run_id='dry',
                        groups=['l1'], prefix_lengths=[2048, 4096], suffix_lengths=[128, 256],
                        chunked_prefill_size=4096)
        return Job(argparse.Namespace(**(settings | changes)), environ={
            'SGLANG_SIMULATE_UNIFORM_EXPERTS': '1'})

    def test_defaults_and_cli_overrides_generate_only_formal_buckets(self):
        job = self.job()
        self.assertEqual(job.resolution['server']['load-format'], 'dummy')
        self.assertEqual(job.resolution['server']['cuda-graph-bs-prefill'], [128, 256])
        env = job.resolution['environment']
        self.assertEqual(env['SGLANG_SIMULATE_UNIFORM_EXPERTS'], '0')
        self.assertEqual(env['SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS'], '1')
        self.assertEqual(self.job(groups=['cold', 'l1']).resolution['server']['cuda-graph-bs-prefill'],
                         [128, 256, 2176, 2304, 4096])
        other = self.job(load_format='auto', routing='original', prefill_graph='disabled')
        self.assertEqual(other.resolution['server']['load-format'], 'auto')
        self.assertNotIn('cuda-graph-bs-prefill', other.resolution['server'])
        self.assertEqual(other.resolution['environment']['SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS'], '0')

    def test_dummy_does_not_inspect_even_broken_existing_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'config.json').write_text('{}')
            (path / 'tokenizer.json').write_text('{}')
            (path / 'model.safetensors.index.json').write_text('broken')
            with patch('hicache.prepare.inspect_model', side_effect=AssertionError('read weights')):
                result = prepare_model('org/model', 'a'*40, path, load_format='dummy')
            self.assertEqual(result['weight_bytes'], 0)
            self.assertFalse(result['download_requested'])

    def test_dummy_download_is_explicitly_filtered(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            def download(**options):
                self.assertEqual(options['allow_patterns'], list(ASSET_FILES))
                self.assertTrue(all(not name.endswith(('.bin', '.safetensors')) and 'index' not in name
                                    for name in options['allow_patterns']))
                (path / 'config.json').write_text('{}')
                (path / 'tokenizer.json').write_text('{}')
                return temp
            self.assertEqual(prepare_model('org/model', 'a'*40, downloader=download,
                                          load_format='dummy')['load_format'], 'dummy')

    def test_compact_input_reconstructs_identical_requests(self):
        class Tokenizer:
            all_special_ids = []
            def encode(self, text, **_): return list(text.encode())
        tokenizer = Tokenizer()
        cases = benchmark.make_cases(tokenizer, [2048, 4096], [128, 256], 'fixed')
        saved = benchmark.input_manifest(tokenizer, cases, 'fixed')
        for case in saved['cases']:
            ids = (benchmark.tile(saved['prefix_unit'], case['prefix_length']) +
                   benchmark.tile(saved['suffix_unit'], case['suffix_length']))
            self.assertEqual(benchmark.digest(ids), case['input_sha256'])
        self.assertLess(len(json.dumps(saved)), len(json.dumps(cases)) // 5)


class GraphEvidence(unittest.TestCase):
    def prometheus(self, mode):
        lines = []
        for rank in range(2):
            labels = f'tp_rank="{rank}",pp_rank="0",dp_rank="0",moe_ep_rank="{rank}"'
            for (name, label, value), key in metrics.FORWARD.items():
                if key in ('graph_passes', 'eager_passes') and key != mode + '_passes':
                    continue
                lines.append(f'{name}{{{labels},{label}="{value}"}} 1')
            for name in metrics.LOAD:
                lines.append(f'{name}{{{labels}}} 0')
            lines.append(f'sglang:load_back_tokens_total{{{labels},pool="kv"}} 0')
        return '\n'.join(lines)

    def test_lazy_counter_labels_and_each_rank_are_checked(self):
        for mode, wanted in [('graph', 'breakable'), ('eager', 'disabled')]:
            parsed = metrics.parse_metrics(self.prometheus(mode), 2, 2, False)
            self.assertEqual(parsed[0]['passes'], 1)
            self.assertEqual(parsed[0][mode+'_passes'], 1)
            self.assertFalse(metrics.graph_error(parsed, wanted))
            parsed[1].update(graph_passes=0, eager_passes=1)
            if wanted == 'breakable':
                self.assertIn('rank=1', metrics.graph_error(parsed, wanted))
        with self.assertRaises(metrics.PendingMetrics):
            metrics.parse_metrics(self.prometheus('graph').splitlines()[0], 2, 2, False)

    def test_graph_fallback_is_saved_then_stops_sampling(self):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=('case', 'group', 'attempt', 'valid', 'invalid_reason',
                                                   'client_latency_ms', 'output_ids'))
        writer.writeheader()
        args = argparse.Namespace(seed=0, groups=['l1'], repeats=2, max_attempts=6,
                                  warmup_rounds=0, output_length=1)
        row = dict(case='p', group='l1', valid=False, invalid_reason='[Graph] rank=1 fallback',
                   client_latency_ms=1., output_ids='[1]')
        with patch('hicache.benchmark.execute_sample', return_value=row) as sample:
            with self.assertRaisesRegex(RuntimeError, 'Graph'):
                benchmark.run_samples(argparse.Namespace(rank_metrics=None), [{'id': 'p'}], args,
                                      writer, stream, {})
        self.assertEqual(sample.call_count, 1)
        self.assertIn('False,[Graph]', stream.getvalue())
