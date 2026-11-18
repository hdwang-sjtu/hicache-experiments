"""V4 的协议/缓存边界及打包权重 CPU 检查；不冒充 H200 GPU 验收。"""
import argparse
from array import array
from contextlib import redirect_stdout
from copy import deepcopy
from enum import Enum, auto
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from hicache import metrics
from hicache.benchmark import check_reused_prefix
from hicache.runner import Job
import test_prefix_reuse as reuse_fixture
from patch_functions import added_function


class V4Configuration(unittest.TestCase):
    def job(self, **changes):
        return Job(argparse.Namespace(experiment='deepseek-v4-flash', runtime='native', run_id='dry', **changes), environ={})

    def test_small_grid_and_large_grid_use_existing_entry(self):
        job = self.job()
        server, env = job.resolution['server'], job.resolution['environment']
        self.assertFalse(server['enable-hierarchical-cache'])
        self.assertTrue(server['enable-hicache-experiment'])
        self.assertEqual((server['tp-size'], server['ep-size']), (2, 1))
        self.assertEqual(server['cuda-graph-bs-prefill'], [128, 1024, 2176, 3072, 4224, 5120])
        self.assertEqual(env['SGLANG_DSV4_FP4_EXPERTS'], '1')
        large = self.job(prefix_lengths=[131072], suffix_lengths=[16384], groups=['l1'])
        self.assertGreaterEqual(large.resolution['server']['context-length'], 147457)
        self.assertEqual(large.resolution['server']['cuda-graph-bs-prefill'], [16384])
        eager = self.job(prefill_graph='disabled')
        self.assertNotIn('cuda-graph-bs-prefill', eager.resolution['server'])

    def test_reject_unsupported_or_insufficient_settings_before_start(self):
        for options in (dict(groups=['l2']), dict(ep=2), dict(prefix_lengths=[257]),
                        dict(context_length=2048), dict(suffix_lengths=[65536])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.job(**options)


class DeviceInspection(unittest.TestCase):
    def setUp(self):
        class Component(Enum):
            FULL = auto()
            SWA = auto()
        class Core:
            pass
        self.full = NS(size=4096, available_size=lambda: 3584)
        self.swa = NS(size=1024, available_size=lambda: 512)
        self.kv = NS(compression_ratios=[0, 4, 128])
        allocator = NS(full_attn_allocator=self.full, swa_attn_allocator=self.swa,
                       get_kvcache=lambda: self.kv)
        self.match = Mock(return_value=NS(device_prefix_len=512, full_kv_hit_length=512))
        self.cache = NS(disable=False, tree_core=Core(), tree_components=[Component.FULL, Component.SWA],
                        sidecar_pool_specs=[], session=NS(slots={}), linker=None, enable_storage=False,
                        page_size=256, token_to_kv_pool_allocator=allocator,
                        req_to_token_pool=NS(available_size=lambda: 8), match_prefix=self.match)
        self.inspect = added_function('_experiment_device_state', dict(
            UnifiedTreeCore=Core, ComponentType=Component, array=array,
            MatchPrefixParams=lambda **kwargs: NS(**kwargs), RadixKey=lambda ids: ids))

    def test_inspect_without_host_controller_uses_real_component_match(self):
        state = self.inspect(self.cache, [7]*512, action='inspect')
        self.assertEqual(state['cache_mode'], 'device')
        self.assertEqual(state['device_used_tokens'], 512)
        self.assertFalse(state['backup_complete'])
        self.assertEqual(state['pool_ownership']['c4_state'], 'swa')
        metrics.check_state(state, 'l1', 512, 1)
        # FULL KV 完整但缺 SWA 的更短匹配，绝不能被当成命中完整 P。
        self.match.return_value.device_prefix_len = 256
        state = self.inspect(self.cache, [7]*512, action='inspect')
        with self.assertRaises(RuntimeError):
            metrics.check_state(state, 'l1', 512, 1)

    def test_demote_and_unverified_layouts_fail_before_matching(self):
        with self.assertRaisesRegex(ValueError, 'demote'):
            self.inspect(self.cache, [7]*512, action='demote')
        self.match.assert_not_called()
        for changes in (dict(_unified_kv=True), dict(compression_ratios=[2])):
            self.kv.__dict__.update(changes)
            with self.assertRaises(ValueError):
                self.inspect(self.cache, [7]*512, action='inspect')
            self.kv.__dict__.pop('_unified_kv', None)

    def test_reuse_inspects_full_input_with_a_partial_suffix_page(self):
        prefix, suffix = [7]*2048, [8]*128  # V4 默认网格中的第一个点。
        self.match.return_value.device_prefix_len = len(prefix)
        baseline = self.inspect(self.cache, prefix, action='inspect')
        client = NS(rank_metrics=NS(tp=1),
                    wait_cache=lambda ids: self.inspect(self.cache, ids, action='inspect'))
        check_reused_prefix(client, dict(prefix=prefix, suffix=suffix), baseline, baseline)
        self.assertEqual(len(self.match.call_args.args[0].key), len(prefix) + len(suffix))


class DeviceMetrics(unittest.TestCase):
    def prometheus(self, *, omit_rank=None):
        lines = []
        values = dict(seconds=.1, eager_passes=0, graph_passes=1,
                      compute=128, input=128, device=512, host=0, storage=0)
        for rank in range(2):
            if rank == omit_rank:
                continue
            labels = f'tp_rank="{rank}",pp_rank="0",dp_rank="0",moe_ep_rank="0"'
            for (name, label, value), key in metrics.FORWARD.items():
                lines.append(f'{name}{{{labels},{label}="{value}"}} {values[key]}')
        return '\n'.join(lines)

    def test_forward_only_evidence_needs_both_ranks_but_no_h2d_metrics(self):
        after = metrics.parse_metrics(self.prometheus(), 2, 1, False, hierarchical=False)
        self.assertNotIn('load_seconds', after[0])
        before = {r: dict.fromkeys(v, 0) for r, v in after.items()}
        state = {'ranks': {str(r): dict(cache_mode='device', device_tokens=512,
                                      host_hit_tokens=0, host_backed_tokens=0) for r in range(2)}}
        args = dict(p=512, s=128, group='l1', chunk=32768, state=state, hierarchical=False)
        delta = metrics.metric_delta(before, after, **args)
        self.assertEqual(metrics.graph_error(delta, 'breakable'), '')
        with self.assertRaises(metrics.PendingMetrics):
            metrics.parse_metrics(self.prometheus(omit_rank=1), 2, 1, False, hierarchical=False)
        with self.assertRaises(RuntimeError):
            metrics.metric_delta(before, after, **(args | dict(group='l2')))
        broken = deepcopy(after)
        broken[1]['compute'] = 129
        with self.assertRaises(RuntimeError):
            metrics.metric_delta(before, broken, **args)

    def test_prefix_reuse_has_no_host_backup_and_rejects_one_rank_swa_leak(self):
        class DeviceCache(reuse_fixture.Cache):
            hicache = False
            def wait_cache(self, ids, *, demote=False):
                state = super().wait_cache(ids, demote=demote)
                for rank, row in state['ranks'].items():
                    row.update(cache_mode='device', cache_components=['full', 'swa', 'c4', 'c128'],
                               host_hit_tokens=0, host_backed_tokens=0, host_used_tokens=0,
                               backup_complete=False,
                               device_pools=dict(full=1024-len(self.cached), swa=100, request=8))
                    if self.corrupt and self.fault == 'swa' and rank == '3':
                        row['device_pools']['swa'] -= 1
                state.update(state['ranks']['0'])
                return state
        fixture = reuse_fixture.PrefixReuse()
        fixture.setUp()
        args = deepcopy(fixture.args)
        args.groups = ['cold', 'l1']
        with redirect_stdout(io.StringIO()):
            rows = fixture.run_cache(DeviceCache(), args)
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row['valid'] == 'True' for row in rows))
            with self.assertRaisesRegex(RuntimeError, 'Prefix reuse'):
                fixture.run_cache(DeviceCache(fault='swa'), args)

    def test_reader_archives_forward_evidence_without_fabricating_h2d_measurements(self):
        after = metrics.parse_metrics(self.prometheus(), 2, 1, False, hierarchical=False)
        before = {r: dict.fromkeys(v, 0) for r, v in after.items()}
        client = NS(timeout=2, request=lambda *a, **kw: (self.prometheus(), 1.))
        now = [0.]
        clock = NS(monotonic=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        state = {'ranks': {str(r): dict(cache_mode='device') for r in range(2)}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'rank-metrics.jsonl'
            reader = metrics.RankMetrics(client, dict(tp_size=2, ep_size=1,
                chunked_prefill_size=32768, enable_hierarchical_cache=False), False, path)
            with patch.object(metrics, 'time', clock):
                result = reader.finish(before, 'measure', 'request', p=512, s=128, group='l1', state=state)
            self.assertEqual(result['prefill_forward_gpu_ms'], 100.)
            self.assertNotIn('kv_load_ms', result)
            record = json.loads(path.read_text())
            self.assertFalse(record['expected']['hierarchical'])
            self.assertEqual(record['deltas']['1']['compute'], 128)
            self.assertGreaterEqual(now[0], .4)


@unittest.skipUnless(importlib.util.find_spec('torch'), '张量检查需要 CPU Torch；可在固定镜像内运行')
class PackedWeightsAndRouting(unittest.TestCase):
    def test_packed_fp4_initialization_is_finite_nonzero_and_seeded(self):
        import torch
        initialize = added_function('initialize_dummy_mxfp4', dict(torch=torch))
        for dtype in (torch.int8, torch.uint8):
            weight = torch.empty(256, 32, dtype=dtype)
            scale = torch.empty(256, 2, dtype=torch.float8_e8m0fnu)
            initialize(weight, scale, 1234)
            expected = weight.clone()
            initialize(weight, scale, 1234)
            self.assertTrue(torch.equal(weight, expected))
            self.assertTrue(torch.equal(scale.view(torch.uint8), torch.full((256, 2), 115, dtype=torch.uint8)))
            codes = weight.view(torch.uint8).long()
            table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.])
            dequant = torch.stack([table[codes & 15], table[codes >> 4]], -1) * 2**-12
            self.assertTrue(torch.isfinite(dequant).all())
            self.assertGreater(int(torch.count_nonzero(dequant)), 0)
            initialize(weight, scale, 1235)
            self.assertFalse(torch.equal(weight, expected))

    def test_shared_routing_helper_preserves_shared_expert_and_scales(self):
        import torch
        calls = []
        def override(ids, weights, num_experts, **options):
            calls.append(options)
            # CPU reference for the existing Triton assignment, EP=1.
            k = ids.shape[1]
            ids.copy_((torch.arange(ids.numel()).reshape_as(ids)+options['layer_id']) % num_experts)
            weights.fill_(options['weight_scale']/k)
        simulate = added_function('_simulate_routed_experts', dict(
            get_parallel=lambda: NS(moe_ep_size=1),
            is_moe_input_scattered_across_dp_ranks=lambda: False,
            _simulate_balanced_routing=override))
        ids = torch.full((1024, 7), 256, dtype=torch.int32)
        weights = torch.full((1024, 7), .75)
        simulate(ids, weights, 256, num_routed_topk=6, layer_id=2, random=False, weight_scale=1.5)
        self.assertTrue((ids[:, -1] == 256).all())
        self.assertTrue((weights[:, -1] == .75).all())
        self.assertTrue((weights[:, :6] == .25).all())
        counts = torch.bincount(ids[:, :6].flatten().long(), minlength=256)
        self.assertEqual(int(counts.max()-counts.min()), 0)
        cold = ids.clone()
        simulate(ids, weights, 256, num_routed_topk=6, layer_id=2, random=False)
        # P/chunk 都按 256 对齐，轮转周期 128，因此 warm hit 的 S 相位相同。
        self.assertTrue(torch.equal(cold[:128, :6], ids[256:384, :6]))
        self.assertEqual(calls[0]['num_token_shards'], 1)


if __name__ == '__main__':
    unittest.main()
