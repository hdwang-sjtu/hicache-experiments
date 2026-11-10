"""跨 S 复用的协议回归；假缓存不代替 GPU 上的 GDN/allocator 验收。"""
import argparse
from collections import Counter
import csv
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hicache import benchmark
from hicache.config import build_run_parser
from hicache.results import FIELDS
from hicache.runner import test_arguments, verify_results


class Reader:
    tp = ep = 4
    gpu_timing = False

    def snapshot(self, *_args):
        return {}

    def finish(self, *_args, **_kwargs):
        return {}


class Cache:
    hicache = True
    rank_metrics = Reader()

    def __init__(self, *, unsupported=False, fault=None):
        self.cached, self.backed, self.calls = [], [], []
        self.unsupported, self.fault = unsupported, fault
        self.corrupt = False

    def flush(self):
        self.cached, self.backed = [], []

    @staticmethod
    def shared(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

    def wait_cache(self, ids, *, demote=False):
        if demote:
            self.cached = []
        device, host = self.shared(ids, self.cached), self.shared(ids, self.backed)
        ranks = {}
        for rank in range(4):
            ranks[str(rank)] = dict(
                skip_radix_cache_insert=not (self.unsupported and rank == 3),
                device_tokens=device, host_hit_tokens=max(host-device, 0),
                host_backed_tokens=host, matched_tokens=max(device, host),
                backup_complete=host == len(ids), device_free_tokens=1024-len(self.cached),
                device_used_tokens=len(self.cached), host_used_tokens=len(self.backed),
                cache_components=['full', 'mamba'],
                mamba_device_states=int(device == len(ids)),
                mamba_host_states=int(host == len(ids)),
                mamba_device_free=64-int(bool(self.cached)), mamba_host_free=64-int(bool(self.backed)),
            )
        if self.corrupt and self.fault == 'slot':
            ranks['3']['mamba_device_free'] -= 1
        return {**ranks['0'], 'ranks': ranks}

    def generate(self, ids, output_length, stage, *, skip_cache_insert=False):
        device, host = self.shared(ids, self.cached), self.shared(ids, self.backed)
        self.calls.append((stage, tuple(ids), skip_cache_insert))
        # A host hit restores P; refusing publication keeps the original tree.
        if skip_cache_insert:
            self.cached = self.backed.copy()
        else:
            self.cached, self.backed = ids.copy(), ids.copy()
        if skip_cache_insert and stage == 'measure':
            self.corrupt = True
            if self.fault == 'suffix':
                self.cached, self.backed = ids.copy(), ids.copy()
        return dict(output_ids=[sum(ids) % 97] * output_length, meta_info=dict(
            prompt_tokens=len(ids), completion_tokens=output_length,
            cached_tokens=max(device, host),
            cached_tokens_details=dict(device=device, host=max(host-device, 0), storage=0),
            num_retractions=0, finish_reason=dict(type='length'),
            e2e_latency=.002, first_token_latency=.001, queue_time=0,
        )), 2., str(len(self.calls))


class PrefixReuse(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(id=f'p{p}_s{s}', prefix=list(range(p)), suffix=[99]*s,
                           input_sha256='test') for p in (8, 16) for s in (2, 4)]
        self.args = argparse.Namespace(seed=0, groups=['cold', 'l1', 'l2'], prefix_reuse=True,
                                       warmup_rounds=1, output_length=1, repeats=3, max_attempts=3,
                                       prefix_lengths=[8, 16], suffix_lengths=[2, 4],
                                       no_plot=True, gpu_timing=False)

    def run_cache(self, client, args=None):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        self.stream = stream
        benchmark.run_samples(client, self.cases, args or self.args, writer, stream, {})
        return list(csv.DictReader(io.StringIO(stream.getvalue())))

    def test_mixed_groups_prepare_once_per_p_and_reuse_across_s(self):
        client = Cache()
        rows = self.run_cache(client)
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(r['valid'] == 'True' for r in rows))
        self.assertEqual(Counter(len(ids) for stage, ids, _ in client.calls if stage == 'prepare'),
                         {8: 1, 16: 1})
        self.assertTrue(all((r['prefix_reuse'] == 'True') == (r['group'] != 'cold') for r in rows))
        for p in (8, 16):
            order = [r['group'] for r in rows if int(r['prefix_length']) == p]
            self.assertEqual(order[:6], ['cold']*6)
            self.assertNotIn('cold', order[6:])

    def test_l1_only_collects_all_references_before_one_prepare(self):
        args = deepcopy(self.args)
        args.groups = ['l1']
        client = Cache()
        self.assertEqual(len(self.run_cache(client, args)), 12)
        stages = [stage for stage, _, _ in client.calls]
        self.assertEqual(stages[:3], ['reference', 'reference', 'prepare'])
        self.assertEqual(stages.count('prepare'), 2)
        self.assertTrue(all(skip for stage, _, skip in client.calls if stage in ('warmup', 'measure')))

    def test_disabled_rebuilds_each_hit_and_never_sends_skip(self):
        args = deepcopy(self.args)
        args.prefix_reuse = False
        client = Cache()
        rows = self.run_cache(client, args)
        self.assertEqual(len(rows), 36)
        self.assertEqual(sum(stage == 'prepare' for stage, _, _ in client.calls), 32)
        self.assertFalse(any(skip for _, _, skip in client.calls))

    def test_unsupported_rank_fails_before_any_generation(self):
        client = Cache(unsupported=True)
        with self.assertRaisesRegex(ValueError, '不支持'):
            self.run_cache(client)
        self.assertEqual(client.calls, [])

    def test_suffix_pollution_and_one_rank_slot_leak_are_saved_and_fatal(self):
        args = deepcopy(self.args)
        args.groups = ['l1']
        for fault in ('suffix', 'slot'):
            with self.subTest(fault=fault), self.assertRaisesRegex(RuntimeError, 'Prefix reuse'):
                self.run_cache(Cache(fault=fault), args)
            rows = list(csv.DictReader(io.StringIO(self.stream.getvalue())))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['valid'], 'False')

    def test_cli_yaml_and_http_flag(self):
        parser, _ = build_run_parser()
        base = ['--tokenizer', '/model', '--output-dir', '/results']
        self.assertTrue(parser.parse_args(base).prefix_reuse)
        self.assertFalse(parser.parse_args(base+['--no-prefix-reuse']).prefix_reuse)
        argv, args = test_arguments({'prefix-reuse': False})
        self.assertIn('--no-prefix-reuse', argv)
        self.assertFalse(args.prefix_reuse)
        client = benchmark.Client('http://localhost', 1, io.StringIO())
        with patch.object(client, 'request', return_value=({}, 1.)) as request:
            client.generate([1, 2], 1, 'measure', skip_cache_insert=True)
            self.assertTrue(request.call_args.args[1]['skip_radix_cache_insert'])

    def test_total_verification_rejects_an_extra_prepare_in_actual_events(self):
        client = Cache()
        self.run_cache(client)
        events = [dict(path='/generate', stage=stage,
                       response=dict(meta_info=dict(prompt_tokens=len(ids))))
                  for stage, ids, _ in client.calls if stage == 'prepare']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'config.json').write_text(json.dumps(dict(status='complete')))
            (root/'inputs.json').write_text('{}')
            (root/'summary.csv').write_text('summary\n')
            (root/'samples.csv').write_text(self.stream.getvalue())
            path = root/'responses.jsonl'
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            self.assertEqual(verify_results(root, self.args)['valid_samples'], 36)
            with path.open('a') as stream:
                stream.write(json.dumps(events[0])+'\n')
            with self.assertRaisesRegex(RuntimeError, '只准备一次'):
                verify_results(root, self.args)


if __name__ == '__main__':
    unittest.main()
