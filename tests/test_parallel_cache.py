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
from urllib.error import HTTPError, URLError

import hicache.metrics as metrics
from hicache.runner import Job
from hicache.benchmark import Client

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
    def test_delayed_rank_publication_and_h2d_only_use_same_reader(self):
        """实际轮询/归档路径：慢 rank 未提交时继续等待，完整后稳定再返回。"""
        for gpu_timing in (True, False):
            with self.subTest(gpu_timing=gpu_timing), tempfile.TemporaryDirectory() as temp:
                after = self.counters()
                before = {r: dict.fromkeys(v, 0) for r, v in after.items()}
                if not gpu_timing:
                    keys = {'kv_tokens', 'load_seconds', 'load_ops', 'load_bytes', 'mamba_states'}
                    before = {r: {k: v for k, v in row.items() if k in keys} for r, row in before.items()}
                    after = {r: {k: v for k, v in row.items() if k in keys} for r, row in after.items()}

                def prometheus(counters):
                    lines = []
                    for rank, row in counters.items():
                        labels = f'tp_rank="{rank}",pp_rank="0",dp_rank="0",moe_ep_rank="{rank}"'
                        for (name, label, value), key in metrics.FORWARD.items():
                            source = 'passes' if key == 'eager_passes' else key
                            if source in row:
                                lines.append(f'{name}{{{labels},{label}="{value}"}} {row[source]}')
                        for name, key in metrics.LOAD.items():
                            lines.append(f'{name}{{{labels}}} {row[key]}')
                        for pool, key in [('kv', 'kv_tokens'), ('mamba', 'mamba_states')]:
                            lines.append(f'sglang:load_back_tokens_total{{{labels},pool="{pool}"}} {row[key]}')
                    return '\n'.join(lines)

                moments, polls = [0.0], []
                def request(*_, **__):
                    polls.append(moments[0])
                    # 首次只有 rank 0 完成；即使 HTTP 已返回，也不能收为最终证据。
                    value = {0: after[0], 1: before[1]} if len(polls) == 1 else after
                    return prometheus(value), 1.0
                client = Mock(timeout=2, request=request)
                clock = Mock(monotonic=lambda: moments[0], sleep=lambda dt: moments.__setitem__(0, moments[0]+dt))
                path = Path(temp) / 'ranks.jsonl'
                reader = metrics.RankMetrics(client, dict(tp_size=2, ep_size=2, chunked_prefill_size=256),
                                             True, path, gpu_timing=gpu_timing)
                with patch.object(metrics, 'time', clock):
                    result = reader.finish(before, 'measure', 'request-1',
                                           p=256, s=128, group='l2', state=self.state())
                self.assertGreaterEqual(polls[-1] - polls[1], .4)
                self.assertEqual(result['kv_load_bytes'], 12288)
                self.assertEqual(result['kv_load_tokens'], 256)
                self.assertEqual('prefill_forward_gpu_ms' in result, gpu_timing)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['after']['1'], after[1])

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
        sample = dict(seconds=.1, passes=1, eager_passes=1, graph_passes=0,
                      compute=128, input=128, device=0, host=256, storage=0,
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


class ClientArchive(unittest.TestCase):
    def test_repeated_metrics_preserve_values_without_growing_archive(self):
        labels = 'tp_rank="0",pp_rank="0",dp_rank="0",moe_ep_rank="0"'
        lines = ['# ' + 'x' * 160_000]
        for name, label, value in metrics.FORWARD:
            lines.append(f'{name}{{{labels},{label}="{value}"}} 0')
        for name in metrics.LOAD:
            lines.append(f'{name}{{{labels}}} 0')
        for pool in ('kv', 'mamba'):
            lines.append(f'sglang:load_back_tokens_total{{{labels},pool="{pool}"}} 0')
        text = '\n'.join(lines)
        events = io.StringIO()
        client = Client('http://localhost:39081', 1, events)
        with patch('hicache.benchmark.urllib.request.urlopen',
                   side_effect=lambda *_args, **_kwargs: io.BytesIO(text.encode())):
            for _ in range(32):
                result, elapsed = client.request('/metrics', stage='metrics_before', parse_json=False)
                self.assertEqual(result, text)
                self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(events.getvalue(), '')
        self.assertEqual(metrics.parse_metrics(result, 1, 1, True)[0]['seconds'], 0)

        # 大量轮询之后，生成响应和缓存控制响应仍然保留原始内容与请求关联。
        for path, payload, response in [
            ('/generate', {'rid': 'measured-request'}, {'output_ids': [7]}),
            ('/debug/hicache', {'action': 'inspect'}, {'success': True, 'ready': True}),
        ]:
            with patch('hicache.benchmark.urllib.request.urlopen',
                       return_value=io.BytesIO(json.dumps(response).encode())):
                result, _ = client.request(path, payload, stage='measure')
            record = json.loads(events.getvalue().splitlines()[-1])
            self.assertEqual(record['path'], path)
            self.assertEqual(record['request_id'], payload.get('rid'))
            self.assertEqual(record['response'], result)
            self.assertEqual(result, response)
        self.assertEqual(len(events.getvalue().splitlines()), 2)

    def test_metrics_failures_are_still_archived_and_fatal(self):
        cases = [
            (HTTPError('http://localhost:39081/metrics', 500, 'failure', {}, io.BytesIO(b'metrics failed')),
             'metrics failed'),
            (URLError('connection refused'), ''),
            (TimeoutError('timed out'), ''),
        ]
        for error, body in cases:
            with self.subTest(error=type(error).__name__):
                events = io.StringIO()
                client = Client('http://localhost:39081', 1, events)
                with patch('hicache.benchmark.urllib.request.urlopen', side_effect=error), \
                     self.assertRaisesRegex(RuntimeError, 'metrics_after /metrics'):
                    client.request('/metrics', stage='metrics_after', parse_json=False)
                record = json.loads(events.getvalue())
                self.assertEqual(record['path'], '/metrics')
                self.assertEqual(record['stage'], 'metrics_after')
                self.assertIn(str(error), record['error'])
                self.assertEqual(record['error_body'], body)


class GroupSelection(unittest.TestCase):
    def test_l1_only_still_has_cold_reference_outside_csv(self):
        import hicache.benchmark as driver
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
        from hicache.prepare import inspect_model
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
