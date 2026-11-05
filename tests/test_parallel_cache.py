"""Distributed agreement, delayed metrics, checkpoint and capacity failure cases."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import io
import csv
from pathlib import Path
import tempfile
import struct
import threading
import unittest
from unittest.mock import Mock, patch

import rank_metrics as metrics
from run_all import Job

PROJECT = Path(__file__).resolve().parents[1]


def coordinator():
    # Test the exact function distributed in the patch, without importing CUDA.
    lines = (PROJECT / 'patches/sglang-hicache-experiment.patch').read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('+def coordinate_hicache_experiment('))
    added = []
    for line in lines[index:]:
        if not line.startswith('+'):
            break
        added.append(line[1:])
    tree = ast.parse('\n'.join(added))
    scope = {}
    exec(compile(tree, '<bundled scheduler patch>', 'exec'), scope)
    return scope['coordinate_hicache_experiment']


class DistributedControl(unittest.TestCase):
    def scenario(self, *, busy=False, fail_prepare=False, fail_execute=False):
        function = coordinator()
        barrier = threading.Barrier(2, timeout=3)
        exchange, actions = {}, []
        def worker(rank):
            phase = 0
            def gather(value):
                nonlocal phase
                exchange[phase, rank] = deepcopy(value)
                barrier.wait()
                values = [deepcopy(exchange[phase, r]) for r in range(2)]
                barrier.wait()
                phase += 1
                return values
            def control(ids, action):
                actions.append((rank, action))
                if rank == 1 and ((fail_prepare and action == 'preflight_demote') or
                                  (fail_execute and action == 'demote')):
                    raise RuntimeError('injected failure')
                return dict(device_tokens=0, host_hit_tokens=len(ids))
            cache = Mock(experiment_control=control)
            return function(cache, [1] * 256, 'demote', not (busy and rank == 1), rank, gather)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(worker, range(2)))
        return result, actions

    def test_all_ranks_preflight_before_any_free(self):
        result, actions = self.scenario()
        self.assertTrue(all(r['success'] and r['ready'] for r in result))
        self.assertEqual([a for _, a in actions[:2]], ['preflight_demote'] * 2)
        self.assertEqual(set(result[0]['state']['ranks']), {'0', '1'})

    def test_busy_or_failed_rank_prevents_all_frees(self):
        for option in ('busy', 'fail_prepare'):
            result, actions = self.scenario(**{option: True})
            self.assertFalse(any(a == 'demote' for _, a in actions))
            self.assertTrue(all(not r['ready'] if option == 'busy' else not r['success'] for r in result))

    def test_partial_execution_failure_is_fatal_on_every_rank(self):
        result, _ = self.scenario(fail_execute=True)
        self.assertTrue(all(not r['success'] and '必须停止' in r['message'] for r in result))


class RankEvidence(unittest.TestCase):
    def state(self, tp=2):
        return {'ranks': {str(r): dict(device_tokens=0, host_hit_tokens=256, host_backed_tokens=256,
                                      backup_complete=True, cache_components=['full', 'mamba'],
                                      mamba_device_states=0, mamba_host_states=1,
                                      kv_bytes_per_token=16, mamba_state_bytes=1024) for r in range(tp)}}

    def test_every_rank_must_have_complete_checkpoint(self):
        state = self.state()
        metrics.check_state(state, 'l2', 256, 2)
        state['ranks']['1']['mamba_host_states'] = 0
        with self.assertRaises(RuntimeError): metrics.check_state(state, 'l2', 256, 2)
        with self.assertRaises(RuntimeError): metrics.check_state(self.state(), 'l2', 256, 4)

    def counters(self, tp=2):
        sample = dict(seconds=.1, passes=1, compute=128, input=128, device=0, host=256, storage=0,
                      kv_tokens=256, load_seconds=.01, load_ops=1, load_bytes=6144, mamba_states=2)
        return {r: dict(sample) for r in range(tp)}

    def test_reload_includes_both_pools_and_all_ranks(self):
        after = self.counters()
        before = {r: dict.fromkeys(v, 0) for r, v in after.items()}
        kwargs = dict(p=256, s=128, group='l2', chunk=256, state=self.state())
        self.assertEqual(len(metrics.metric_delta(before, after, **kwargs)), 2)
        after[1]['host'] = 0
        with self.assertRaises(metrics.PendingMetrics): metrics.metric_delta(before, after, **kwargs)
        after[1]['host'] = 257
        with self.assertRaises(RuntimeError): metrics.metric_delta(before, after, **kwargs)
        after = self.counters()
        after[1]['load_bytes'] += 1
        with self.assertRaises(RuntimeError): metrics.metric_delta(before, after, **kwargs)

    def test_cold_chunk_counts_and_l1_no_hidden_reload(self):
        after = self.counters(1)
        after[0].update(compute=640, input=640, host=0, kv_tokens=0, load_seconds=0, load_ops=0,
                        load_bytes=0, mamba_states=0, passes=3)
        before = {0: dict.fromkeys(after[0], 0)}
        metrics.metric_delta(before, after, p=512, s=128, group='cold', chunk=256, state=self.state(1))
        after[0].update(compute=128, input=128, device=512, passes=1, mamba_states=1)
        with self.assertRaises(RuntimeError):
            metrics.metric_delta(before, after, p=512, s=128, group='l1', chunk=256, state=self.state(1))

    def test_metric_labels_missing_and_duplicates(self):
        lines = []
        for r in range(2):
            labels = f'tp_rank="{r}",pp_rank="0",dp_rank="0",moe_ep_rank="{r}"'
            for (name, label, value), _ in metrics.FORWARD.items():
                lines.append(f'{name}{{{labels},{label}="{value}"}} 0')
            for name in metrics.LOAD:
                lines.append(f'{name}{{{labels}}} 0')
            for pool in ('kv', 'mamba'):
                lines.append(f'sglang:load_back_tokens_total{{{labels},pool="{pool}"}} 0')
        text = '\n'.join(lines)
        self.assertEqual(len(metrics.parse_metrics(text, 2, 2, True)), 2)
        with self.assertRaises(metrics.PendingMetrics): metrics.parse_metrics('\n'.join(lines[:-1]), 2, 2, True)
        with self.assertRaises(RuntimeError): metrics.parse_metrics(text+'\n'+lines[0], 2, 2, True)
        with self.assertRaises(RuntimeError): metrics.parse_metrics(text, 2, 1, True)

    def test_capacity_overrides_and_local_model_are_frozen(self):
        args = argparse.Namespace(experiment='qwen35-35b-matrix', runtime='native', run_id='test',
                                  model_dir='/readonly/model', prefix_lengths=[131072], suffix_lengths=[2048],
                                  tp=4, ep=4)
        job = Job(args, environ={})
        server = job.resolution['server']
        self.assertEqual((server['context-length'], server['chunked-prefill-size']), (135168, 32768))
        self.assertEqual(str(job.model), '/readonly/model')
        for changed in (dict(ep=2), dict(context_length=8192), dict(chunked_prefill_size=1024)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                Job(argparse.Namespace(**(vars(args) | changed)), environ={})


class GroupSelection(unittest.TestCase):
    def test_l1_only_still_has_cold_reference_outside_csv(self):
        import run_experiment as driver
        args = argparse.Namespace(seed=0, groups=['l1'], output_length=1,
                                  warmup_rounds=1, repeats=2, max_attempts=2)
        calls = []
        def sample(client, case, group, output_length, stage='measure'):
            calls.append((group, stage))
            return dict(group=group, valid=True, output_ids='[7]', client_latency_ms=1, invalid_reason='')
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=driver.FIELDS)
        writer.writeheader()
        with patch.object(driver, 'execute_sample', side_effect=sample):
            driver.run_samples(argparse.Namespace(rank_metrics=True), [{'id': 'case'}],
                               args, writer, buffer, {})
        self.assertEqual(calls, [('cold', 'reference'), ('l1', 'warmup'), ('l1', 'measure'), ('l1', 'measure')])
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual([r['group'] for r in rows], ['l1', 'l1'])


class LocalModel(unittest.TestCase):
    def test_readonly_inventory_and_corrupt_tensor_bounds(self):
        from prepare_runtime import inspect_model
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text('{"model_type":"qwen3_5"}')
            (root / 'tokenizer.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text(
                json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
            header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            part = root / 'part.safetensors'
            part.write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            original = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir()}
            self.assertIsNone(inspect_model(root)['revision'])
            self.assertEqual(original, {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir()})
            part.write_bytes(struct.pack('<Q', len(header)) + header + b'12')
            with self.assertRaises(ValueError): inspect_model(root)
            part.unlink()
            with self.assertRaises(FileNotFoundError): inspect_model(root)


if __name__ == '__main__': unittest.main()
